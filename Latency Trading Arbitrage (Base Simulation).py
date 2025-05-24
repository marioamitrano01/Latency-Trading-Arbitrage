import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@njit
def calculate_price_disparity(broker_price, exchange_price):
    
    return broker_price - exchange_price

@njit
def calculate_rs(series, lag):
    n_chunks = len(series) // lag
    chunks = np.zeros((n_chunks, lag))
    
    for i in range(n_chunks):
        chunks[i] = series[i*lag:(i+1)*lag]
    
    rs_values = np.zeros(n_chunks)
    for i, chunk in enumerate(chunks):
        mean_adj = chunk - np.mean(chunk)
        
        cumdev = np.cumsum(mean_adj)
        
        r = np.max(cumdev) - np.min(cumdev)
        
        s = np.std(chunk)
        
        if s > 0:
            rs_values[i] = r/s
        else:
            rs_values[i] = 1.0
    
    if len(rs_values) > 0:
        return np.mean(rs_values)
    else:
        return 1.0

class NascentLatencyArbitrage:
    
    
    def __init__(self, initial_capital=10000.0, risk_limit=0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limit = risk_limit  
        
        self.latency_mu = np.log(0.002)   
        self.latency_sigma = 0.5          
        self.latency_prior_alpha = 2.0     
        self.latency_prior_beta = 0.001   
        
        self.prior_samples = np.random.lognormal(self.latency_mu, self.latency_sigma, 10000)
        self.likelihood_memory = 0.8       
        
        self.k_multiplier = 1.5            
        self.min_threshold = 0.0001       
        self.vol_dampening = 0.95          
        
        self.price_kf = self._initialize_dual_state_kalman()
        
        self.omega = 0.000002     
        self.alpha_garch = 0.1     
        self.beta_garch = 0.85    
        self.garch_variance = self.omega / (1 - self.alpha_garch - self.beta_garch)  
        
        self.latency_kf = self._initialize_latency_kalman()
        
        self.source_weights = None        
        self.source_reliability = []      
        self.triangulation_memory = 20    
        
        self.cp_detector = ChangePointDetector(window_size=30, significance=2.5)
        
        self.arima_coeffs = {'ar': [0.2, -0.1], 'ma': [0.1]}
        
        self.hurst_window = 100
        self.hurst_exponent = 0.5
        
        self.trades = []
        self.performance_log = []
        
        self.quote_process = QuoteUpdateProcess(
            lambda_0=2.0,           
            alpha_intensity=5.0,    
            beta_decay=3.0         
        )
        
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.running = False
    
    def _initialize_dual_state_kalman(self):
        
        #State vector: x_t = [price, volatility]^T
        
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        kf.x = np.array([0.0, 0.001])
        
        # State transition matrix (F)
        # [ 1  0 ]  price persists with random walk
        # [ 0  β ]  volatility persists with autoregression (β < 1)
        kf.F = np.array([
            [1.0, 0.0],
            [0.0, 0.97] 
        ])
        
        kf.H = np.array([[1.0, 0.0]])
        
        kf.Q = np.array([
            [0.001, 0.0],   
            [0.0, 0.0001]  
        ])
        
        kf.R = np.array([[0.0001]])  
        
        kf.P = np.array([
            [0.01, 0.0],
            [0.0, 0.001]
        ])
        
        return kf
    
    def _initialize_latency_kalman(self):
        
        kf = KalmanFilter(dim_x=1, dim_z=1)
        
        kf.x = np.array([0.002])
        
        kf.F = np.array([[0.99]])  
        
        kf.H = np.array([[1.0]])
        
        kf.Q = np.array([[0.0000001]])
        
        kf.R = np.array([[0.000001]])
        
        kf.P = np.array([[0.0001]])
        
        return kf
    
    def update_latency_estimate_bayesian(self, observed_latency):
        
        #Update latency estimate using sequential Bayesian inference
        # p(δ|D) ∝ p(D|δ) · p(δ)
        # We model δ ~ LogNormal(μ, σ²) and update parameters
        # through sequential Bayesian inference
        # Calculate likelihood: p(D|δ)
        likelihood = stats.lognorm.pdf(
            self.prior_samples, 
            s=self.latency_sigma,
            scale=np.exp(self.latency_mu)
        )
        
        likelihood = np.nan_to_num(likelihood, nan=1e-10, posinf=1e-10, neginf=1e-10)
        
        if hasattr(self, 'previous_likelihood'):
            likelihood = (
                self.likelihood_memory * self.previous_likelihood + 
                (1 - self.likelihood_memory) * likelihood
            )
        
        self.previous_likelihood = likelihood.copy()
        
        sum_likelihood = np.sum(likelihood)
        if sum_likelihood > 0:
            posterior_weights = likelihood / sum_likelihood
        else:
            posterior_weights = np.ones_like(likelihood) / len(likelihood)
        
        posterior_weights = np.nan_to_num(posterior_weights, nan=1.0/len(posterior_weights))
        
        posterior_weights = posterior_weights / np.sum(posterior_weights)
        
        indices = np.random.choice(
            len(self.prior_samples), 
            size=len(self.prior_samples), 
            p=posterior_weights
        )
        self.prior_samples = self.prior_samples[indices]
        
        self.latency_mu = np.mean(np.log(self.prior_samples))
        self.latency_sigma = np.std(np.log(self.prior_samples))
        
        self.latency_kf.predict()
        self.latency_kf.update(observed_latency)
        
        kalman_latency = self.latency_kf.x[0]
        bayesian_latency = np.exp(self.latency_mu)
        
        kalman_weight = 1.0 / (1.0 + self.latency_kf.P[0, 0] * 1000)
        blended_latency = (
            kalman_weight * kalman_latency + 
            (1 - kalman_weight) * bayesian_latency
        )
        
        return blended_latency, np.exp(self.latency_sigma)
    
    def update_kalman_filter(self, price):
       
        self.price_kf.predict()
        
        self.price_kf.update(price)
        
        filtered_price = self.price_kf.x[0]
        filtered_volatility = abs(self.price_kf.x[1])  
        
        return filtered_price, filtered_volatility
    
    def update_garch_volatility(self, return_t):
       
        self.garch_variance = (
            self.omega + 
            self.alpha_garch * return_t**2 + 
            self.beta_garch * self.garch_variance
        )
        
        return np.sqrt(self.garch_variance)
    
    def calculate_adaptive_threshold(self, kalman_vol, garch_vol, historical_vol=None):
        
        if historical_vol is not None:
            garch_weight = min(garch_vol / (kalman_vol + 1e-10), 0.7)
            kalman_weight = 0.7 - garch_weight
            hist_weight = 0.3
            
            blended_vol = (
                garch_weight * garch_vol + 
                kalman_weight * kalman_vol + 
                hist_weight * historical_vol
            )
        else:
            garch_weight = 0.6
            blended_vol = garch_weight * garch_vol + (1 - garch_weight) * kalman_vol
        
        if hasattr(self, 'previous_threshold'):
            blended_vol = (
                self.vol_dampening * self.previous_blended_vol + 
                (1 - self.vol_dampening) * blended_vol
            )
        
        self.previous_blended_vol = blended_vol
        
        threshold = self.k_multiplier * blended_vol
        
        threshold = max(threshold, self.min_threshold)
        
        self.previous_threshold = threshold
        
        return threshold, blended_vol
    
    def multi_source_triangulation(self, broker_price, exchange_price, other_sources=None):
        
        #Validate price disparities using multiple sources with adaptive weighting
        #Δp_validated(t) = Σ(i=1 to n) w_i · (P_i(t) - P_exchange(t-δ_i))
        
        primary_disparity = calculate_price_disparity(broker_price, exchange_price)
        
        if other_sources is None or len(other_sources) == 0:
            return primary_disparity, 1.0  
        
        disparities = [primary_disparity]
        for source_price in other_sources:
            src_disparity = source_price - exchange_price
            disparities.append(src_disparity)
        
        if self.source_weights is None:
            self.source_weights = np.ones(len(disparities)) / len(disparities)
            self.source_reliability = [[] for _ in range(len(disparities))]
        
        median_disparity = np.median(disparities)
        for i, disparity in enumerate(disparities):
            agreement = 1.0 / (1.0 + abs(disparity - median_disparity))
            

            self.source_reliability[i].append(agreement)
            if len(self.source_reliability[i]) > self.triangulation_memory:
                self.source_reliability[i].pop(0)
        

        new_weights = []
        for i in range(len(disparities)):
            if len(self.source_reliability[i]) > 0:

                weighted_reliability = np.average(
                    self.source_reliability[i],
                    weights=np.exp(np.linspace(0, 1, len(self.source_reliability[i])))
                )
                new_weights.append(weighted_reliability)
            else:
                new_weights.append(1.0)
        

        self.source_weights = np.array(new_weights) / sum(new_weights)
        

        weighted_disparity = np.sum(self.source_weights * disparities)
        

        disparity_std = np.std(disparities)
        entropy = -np.sum(self.source_weights * np.log2(self.source_weights + 1e-10))
        max_entropy = np.log2(len(self.source_weights))
        
        confidence = (
            0.7 * (1.0 / (1.0 + disparity_std * 10)) + 
            0.3 * (entropy / max_entropy)
        )
        
        return weighted_disparity, confidence
    
    def estimate_hurst_exponent(self, price_series, window=None):
        
        if window is None:
            window = self.hurst_window
        
        if len(price_series) < window:
            return 0.5  
        
        series = price_series[-window:]
        
        lags = [2, 4, 8, 16, 32]
        lags = [lag for lag in lags if lag < len(series) // 2]
        
        if len(lags) < 2:
            return 0.5
        
        rs_values = []
        for lag in lags:
            rs = calculate_rs(series, lag)
            rs_values.append(rs)
        
        x = np.log10(lags)
        y = np.log10(rs_values)
        hurst = np.polyfit(x, y, 1)[0]
        
        return hurst
    
    def calculate_edge(self, price_disparity, threshold, transaction_cost=0.0001):
        
        expected_profit = abs(price_disparity)
        
        edge = expected_profit - threshold
        
        edge -= transaction_cost
        
        return edge
    
    def calculate_win_probability(self, price_disparity, volatility, price_history=None):
        
        #Calculate probability of winning based on price disparity and market regime
        #p = CDF(z) where z = price_disparity / (volatility * scaling_factor)
        
        z_score = abs(price_disparity) / (volatility * 1.5)
        
        z_score = min(max(z_score, 0), 3)
        
        base_probability = stats.norm.cdf(z_score)
        
        if price_history is not None and len(price_history) > self.hurst_window:
            hurst = self.estimate_hurst_exponent(price_history)
            
            direction = 1 if price_disparity > 0 else -1
            
            if hurst < 0.45: 
               
                adjustment = 0.1 * (0.5 - hurst) * 2  
                
                adjusted_probability = base_probability + adjustment * direction
            elif hurst > 0.55:  
                
                adjustment = 0.1 * (hurst - 0.5) * 2  # Scale to max ±0.1
                
                adjusted_probability = base_probability - adjustment * direction
            else:
                adjusted_probability = base_probability
            
            win_probability = min(max(adjusted_probability, 0.5), 0.95)
        else:
            win_probability = min(max(base_probability, 0.5), 0.95)
        
        return win_probability
    
    def kelly_criterion(self, edge, win_probability, loss_given_loss=1.0):
        
        if win_probability <= 0.5 or win_probability >= 1 or edge <= 0:
            return 0.0
        
        win_amount = edge
        loss_amount = loss_given_loss
        
        b = win_amount / loss_amount
        
        # Kelly formula
        f_star = (win_probability * b - (1 - win_probability)) / b
        
        #conservative fraction of Kelly (half-Kelly)
        f_star *= 0.5
        
        f_star = min(f_star, self.risk_limit)
        
        return max(0.0, f_star)
    
    async def analyze_opportunity(self, broker_price, exchange_price, price_history, other_sources=None):
        
        current_time = time.time()
        
        filtered_price, kalman_volatility = self.update_kalman_filter(exchange_price)
        
        if len(price_history) > 1:
            returns = np.log(exchange_price / price_history[-1])
            garch_volatility = self.update_garch_volatility(returns)
        else:
            garch_volatility = np.sqrt(self.garch_variance)
        
        if len(price_history) > 30:
            returns = np.diff(np.log(price_history[-30:]))
            historical_volatility = np.std(returns) * np.sqrt(252)  
        else:
            historical_volatility = None
        
        threshold, blended_volatility = self.calculate_adaptive_threshold(
            kalman_volatility, garch_volatility, historical_volatility
        )
        
        raw_disparity = calculate_price_disparity(broker_price, exchange_price)
        
        validated_disparity, confidence = self.multi_source_triangulation(
            broker_price, exchange_price, other_sources
        )
        
        if len(self.performance_log) > self.cp_detector.window_size:
            historical_disparities = [log['price_disparity'] for log in self.performance_log[-self.cp_detector.window_size:]]
            cp_detected, cp_statistic = self.cp_detector.detect(historical_disparities)
        else:
            cp_detected, cp_statistic = False, 0
        
        observed_latency = 0.002 + 0.0005 * np.random.randn()  # ~2ms with noise
        latency_estimate, latency_std = self.update_latency_estimate_bayesian(observed_latency)
        
        edge = self.calculate_edge(validated_disparity, threshold)
        
        win_probability = self.calculate_win_probability(
            validated_disparity, blended_volatility, price_history
        )
        
        kelly_fraction = self.kelly_criterion(edge, win_probability)
        
        intensity = self.quote_process.calculate_intensity(current_time)
        self.quote_process.add_update(current_time)
        
        
        tradable = (
            abs(validated_disparity) > threshold and 
            confidence > 0.7 and
            edge > 0 and
            (cp_detected or abs(validated_disparity) > 1.5 * threshold)
        )
        
        direction = 1 if validated_disparity > 0 else -1 if validated_disparity < 0 else 0
        
        analysis = {
            'timestamp': current_time,
            'broker_price': broker_price,
            'exchange_price': exchange_price,
            'filtered_price': filtered_price,
            'raw_disparity': raw_disparity,
            'validated_disparity': validated_disparity,
            'kalman_volatility': kalman_volatility,
            'garch_volatility': garch_volatility,
            'blended_volatility': blended_volatility,
            'threshold': threshold,
            'confidence': confidence,
            'latency_estimate': latency_estimate * 1000,  
            'latency_std': latency_std * 1000, 
            'edge': edge,
            'win_probability': win_probability,
            'kelly_fraction': kelly_fraction,
            'cp_detected': cp_detected,
            'cp_statistic': cp_statistic,
            'quote_intensity': intensity,
            'tradable': tradable,
            'direction': direction,
            'price_disparity': validated_disparity  
        }
        
        self.performance_log.append(analysis)
        
        return analysis
    
    async def execute_trade(self, opportunity):
        
        if not opportunity['tradable']:
            return None
        
        position_size = self.current_capital * opportunity['kelly_fraction']
        
        expected_profit = position_size * opportunity['edge'] * opportunity['direction']
        
        trade = {
            'timestamp': opportunity['timestamp'],
            'entry_price': opportunity['broker_price'],
            'direction': opportunity['direction'],
            'position_size': position_size,
            'expected_profit': expected_profit,
            'threshold': opportunity['threshold'],
            'volatility': opportunity['blended_volatility'],
            'price_disparity': opportunity['validated_disparity'],
            'edge': opportunity['edge'],
            'win_probability': opportunity['win_probability'],
            'latency_ms': opportunity['latency_estimate'],
            'confidence': opportunity['confidence']
        }
        
        self.trades.append(trade)
        
        logger.info(f"Trade executed: {opportunity['direction']} at {opportunity['broker_price']:.5f}, "
                   f"size: {position_size:.2f}, expected profit: {expected_profit:.2f}")
        
        return trade
    
    async def close_position(self, trade, current_price, hold_time):
        
        if trade['direction'] > 0:  # Long position
            price_change = current_price - trade['entry_price']
        else:  # Short position
            price_change = trade['entry_price'] - current_price
        
        pnl = trade['position_size'] * price_change / trade['entry_price']
        
        pnl -= trade['position_size'] * 0.0001  # 1 basis point
        
        self.current_capital += pnl
        
        trade['exit_price'] = current_price
        trade['hold_time'] = hold_time
        trade['pnl'] = pnl
        trade['roi'] = pnl / trade['position_size']
        
        logger.info(f"Position closed: P&L: {pnl:.2f}, ROI: {trade['roi']:.2%}, New capital: {self.current_capital:.2f}")
        
        return pnl
    
    async def simulate_market_data(self, duration=10800, frequency=0.05, latency_ms=2.0):
        
        #Simulate market data for the latency arbitrage model
        #Simulated time series of prices and metadata
        
        logger.info(f"Simulating market data for {duration/3600:.1f} hours with {latency_ms}ms average latency")
        
        n_steps = int(duration / frequency)
        
        base_price = 15000.0  # Just an example of the base asset price
        price_drift = 0.0    
        price_volatility = 0.0001 
        
        # Convert latency to seconds
        true_latency = latency_ms / 1000.0
        
        timestamps = np.linspace(0, duration, n_steps)
        
        price_changes = np.exp(
            price_drift * frequency + 
            price_volatility * np.random.normal(0, 1, n_steps) * np.sqrt(frequency)
        )
        exchange_prices = base_price * np.cumprod(price_changes)
        
        #occasional jumps
        jump_indices = np.random.choice(n_steps, size=int(n_steps*0.005), replace=False)
        jump_sizes = np.random.normal(0, 0.001, len(jump_indices))
        for idx, jump in zip(jump_indices, jump_sizes):
            exchange_prices[idx:] *= (1 + jump)
        
        vol_cluster_starts = np.random.choice(n_steps, size=int(n_steps*0.01), replace=False)
        vol_cluster_lengths = np.random.randint(10, 100, len(vol_cluster_starts))
        vol_cluster_intensities = np.random.uniform(1.5, 3.0, len(vol_cluster_starts))
        
        for start, length, intensity in zip(vol_cluster_starts, vol_cluster_lengths, vol_cluster_intensities):
            end = min(start + length, n_steps)
            cluster_changes = np.exp(
                price_drift * frequency + 
                price_volatility * intensity * np.random.normal(0, 1, end-start) * np.sqrt(frequency)
            )
            multiplier = np.cumprod(cluster_changes)
            exchange_prices[start:end] *= multiplier / multiplier[0]  
        
        broker_prices = np.zeros_like(exchange_prices)
        
        latency_sigma = 0.5  
        latency_samples = np.random.lognormal(
            np.log(true_latency) - 0.5*latency_sigma**2,  
            latency_sigma, 
            n_steps
        )
        
        spike_period = int(n_steps / 20)  
        spike_indices = np.arange(0, n_steps, spike_period)
        spike_intensities = np.random.uniform(2, 5, len(spike_indices))
        
        for idx, intensity in zip(spike_indices, spike_intensities):
            window = min(50, n_steps - idx)
            latency_samples[idx:idx+window] *= np.linspace(intensity, 1, window)
        
        # Generate broker prices with variable latency
        for i in range(n_steps):
            delay_steps = int(latency_samples[i] / frequency)
            
            source_idx = max(0, i - delay_steps)
            
            broker_prices[i] = exchange_prices[source_idx]
        
        broker_prices *= (1 + np.random.normal(0, 0.00005, n_steps))
        
        n_sources = 3
        other_sources = []
        
        for src in range(n_sources):
            src_latency_base = true_latency * np.random.uniform(0.7, 1.3)
            src_latency_sigma = latency_sigma * np.random.uniform(0.8, 1.2)
            
            src_latency_samples = np.random.lognormal(
                np.log(src_latency_base) - 0.5*src_latency_sigma**2,
                src_latency_sigma,
                n_steps
            )
            
            source_prices = np.zeros_like(exchange_prices)
            for i in range(n_steps):
                delay_steps = int(src_latency_samples[i] / frequency)
                source_idx = max(0, i - delay_steps)
                source_prices[i] = exchange_prices[source_idx]
            
            source_prices *= (1 + np.random.normal(0, 0.0001, n_steps))
            
            other_sources.append(source_prices)
        
        market_data = {
            'timestamps': timestamps,
            'exchange_prices': exchange_prices,
            'broker_prices': broker_prices,
            'other_sources': other_sources,
            'true_latency_samples': latency_samples,
            'metadata': {
                'duration': duration,
                'frequency': frequency,
                'true_latency_ms': latency_ms,
                'base_price': base_price
            }
        }
        
        return market_data
    
    async def backtest(self, duration=10800, latency_ms=2.0):
        
        logger.info(f"Starting backtest of latency arbitrage strategy for {duration/3600:.1f} hours")
        start_time = time.time()
        
        self.current_capital = self.initial_capital
        self.trades = []
        self.performance_log = []
        
        market_data = await self.simulate_market_data(duration=duration, latency_ms=latency_ms)
        
        timestamps = market_data['timestamps']
        exchange_prices = market_data['exchange_prices']
        broker_prices = market_data['broker_prices']
        other_sources_data = market_data['other_sources']
        
        equity_curve = [self.initial_capital]
        open_positions = []
        
        total_trades = 0
        winning_trades = 0
        
        for i in range(1, len(timestamps)):
            current_time = timestamps[i]
            
            exchange_price = exchange_prices[i]
            broker_price = broker_prices[i]
            other_sources = [src[i] for src in other_sources_data]
            
            price_history = exchange_prices[:i]
            
            opportunity = await self.analyze_opportunity(
                broker_price, exchange_price, price_history, other_sources
            )
            
            if opportunity['tradable']:
                trade = await self.execute_trade(opportunity)
                if trade:
                    trade['open_time'] = current_time
                    open_positions.append(trade)
                    total_trades += 1
            
            new_open_positions = []
            for pos in open_positions:
                hold_time = current_time - pos['open_time']
                
                if (hold_time > np.random.uniform(5, 20) or  
                    abs(exchange_price - pos['entry_price'])/pos['entry_price'] > 0.001): 
                    
                    pnl = await self.close_position(pos, exchange_price, hold_time)
                    
                    if pnl > 0:
                        winning_trades += 1
                else:
                    new_open_positions.append(pos)
            
            open_positions = new_open_positions
            
            if i % 20 == 0:  
                # Calculate mark-to-market value of open positions
                open_value = self.current_capital
                for pos in open_positions:
                    if pos['direction'] > 0:  # Long
                        pos_pnl = (exchange_price - pos['entry_price']) * pos['position_size'] / pos['entry_price']
                    else:  # Short
                        pos_pnl = (pos['entry_price'] - exchange_price) * pos['position_size'] / pos['entry_price']
                    open_value += pos_pnl
                
                equity_curve.append(open_value)
        
        for pos in open_positions:
            await self.close_position(
                pos, exchange_prices[-1], timestamps[-1] - pos['open_time']
            )
        
        final_equity = self.current_capital
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = final_equity - self.initial_capital
        profit_pct = (total_profit / self.initial_capital) * 100
        
        # Sharpe ratio (annualized)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 3600 / duration)
        else:
            sharpe = 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_profit': total_profit,
            'profit_pct': profit_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'equity_curve': equity_curve,
            'timestamps': timestamps[::20],  
            'duration_hours': duration / 3600,
            'run_time': time.time() - start_time
        }
        
        logger.info(f"Backtest complete in {results['run_time']:.2f} seconds:")
        logger.info(f"Starting capital: €{self.initial_capital:.2f}")
        logger.info(f"Final capital: €{final_equity:.2f}")
        logger.info(f"Total profit: €{total_profit:.2f} ({profit_pct:.2f}%)")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Sharpe ratio: {sharpe:.2f}")
        
        return results


class ChangePointDetector:
    
    #Implements change-point detection for identifying sub-threshold latency patterns
    
    def __init__(self, window_size=30, significance=2.5):
        self.window_size = window_size
        self.significance = significance  
        self.baseline_mean = 0
        self.baseline_std = 0.0001
        self.history = []
    
    def detect(self, series):
        
        if len(series) < self.window_size:
            return False, 0
        
        self.history.extend(series)
        if len(self.history) > self.window_size * 3:
            self.history = self.history[-self.window_size * 3:]
        
        if len(self.history) > self.window_size:
            baseline = self.history[:-self.window_size]
            self.baseline_mean = np.mean(baseline)
            self.baseline_std = max(np.std(baseline), 0.0001)  
        
        recent = series[-self.window_size:]
        recent_mean = np.mean(recent)
        
        z_score = abs(recent_mean - self.baseline_mean) / self.baseline_std
        
        detected = z_score > self.significance
        
        return detected, z_score


class QuoteUpdateProcess:
    
    #Point process model for broker quote updates
    #λ(t) = λ₀ + Σ(i: t_i<t) α·e^(-β(t-t_i))
    
    def __init__(self, lambda_0=1.0, alpha_intensity=1.5, beta_decay=2.0):
        self.lambda_0 = lambda_0  
        self.alpha_intensity = alpha_intensity  
        self.beta_decay = beta_decay  # Decay rate of intensity
        self.update_times = []  
    
    def add_update(self, t):
        self.update_times.append(t)
        
        if len(self.update_times) > 100:
            self.update_times = self.update_times[-100:]
    
    def calculate_intensity(self, t):
        intensity = self.lambda_0
        
        for t_i in self.update_times:
            if t_i < t:
                intensity += self.alpha_intensity * np.exp(-self.beta_decay * (t - t_i))
        
        return intensity


async def main():
    model = NascentLatencyArbitrage(initial_capital=10000.0)
    
    #backtest for 3 hours
    results = await model.backtest(duration=10800, latency_ms=2.0)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['timestamps'], results['equity_curve'])
    plt.axhline(y=model.initial_capital, color='r', linestyle='--')
    plt.title(f"Latency Arbitrage Performance: €{results['total_profit']:.2f} profit in {results['duration_hours']:.1f} hours")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Account Value (€)")
    plt.grid(True)
    
    if len(model.performance_log) > 0:
        plt.subplot(2, 1, 2)
        
        log_times = [log['timestamp'] for log in model.performance_log]
        disparities = [log['validated_disparity'] for log in model.performance_log]
        thresholds = [log['threshold'] for log in model.performance_log]
        
        plt.plot(log_times, disparities, 'b', alpha=0.7, label='Price Disparity (Δp)')
        plt.plot(log_times, thresholds, 'r--', label='Threshold (τ)')
        plt.plot(log_times, [-t for t in thresholds], 'r--')
        
        for trade in model.trades:
            plt.axvline(x=trade['timestamp'], color='g', alpha=0.3)
        
        plt.title(f"Price Disparities vs Thresholds: {len(model.trades)} trades, {results['win_rate']:.1%} win rate")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Price Difference")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nLatency Arbitrage Strategy Simulation")
    print(f"Initial Capital: €{results['initial_capital']:.2f}")
    print(f"Final Capital: €{results['final_capital']:.2f}")
    print(f"Total Profit: €{results['total_profit']:.2f} ({results['profit_pct']:.2f}%)")
    print(f"Number of Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Runtime: {results['run_time']:.2f} seconds")
    

if __name__ == "__main__":
    asyncio.run(main())

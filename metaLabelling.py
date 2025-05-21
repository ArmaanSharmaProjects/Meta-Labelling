import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('ES_1min_sample.csv')


dollarThreshold = 2000000

def createDollarBars(data, dollarThreshold):
    dollarBarData = []
    dollars = 0
    startIndex = 0

    for i in range(0, len(data)):
        dollars += data['volume'].iloc[i] * data['close'].iloc[i]
        if(dollars > dollarThreshold):
            bar = data.iloc[startIndex : i+1]
            timeStamp = bar['timestamp'].iloc[-1]
            vwap = (bar['close']* bar['volume']).sum()/bar['volume'].sum()
            open = bar['close'].iloc[0]
            high = bar['close'].max()
            low = bar['close'].min()
            close = bar['close'].iloc[-1]
            totalvolume = bar['volume'].sum()
            
            dollarBarData.append([timeStamp, vwap, open, high, low, close, totalvolume])
            dollars = 0
            startIndex = i+1
        
    dollarDf = pd.DataFrame(dollarBarData, columns=['Timestamp', 'VWAP', 'Open', 'High', 'Low', 'Close', 'Total Volume'])
    return dollarDf
 
dollarDf = createDollarBars(data, dollarThreshold=dollarThreshold)
dollarDf.to_csv('tripleBarrierDollarData.csv', index=False)
dollarDf['Timestamp'] = pd.to_datetime(dollarDf['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
dollarDf.set_index('Timestamp', inplace=True)
dollarDf.sort_index(inplace=True) 
close = dollarDf['Close']

#I added series correlation as part of the features
def rolling_autocorr(series, window):
    return series.rolling(window).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )

# I will be using crossing moving averages of the bars to track significant events as well as provide a side. Meta labelling will then be done to decide the size of the bet, or whether to trade

def crossingMovingAverages(data, short_window, long_window):
    df0 = pd.DataFrame(index = data.index)
    df0['short_ma'] = data.rolling(window=short_window).mean()
    df0['long_ma'] = data.rolling(window=long_window).mean()
    df0['ma_diff'] = (df0['short_ma'] - df0['long_ma'])
    df0['serial_corr'] = rolling_autocorr(data.pct_change(), 10)
    df0['volatility'] = data.pct_change().rolling(10).std()
    df0['momentum'] = data.pct_change(5)
    df0['signal'] = 0.0
    df0['signal'][long_window:] = np.where(df0['short_ma'][long_window:] > df0['long_ma'][long_window:], 1.0, 0.0)
    df0['positions'] = df0['signal'].diff()
    df0 = df0[(df0['positions'] == 1) | (df0['positions'] == -1)]
    return df0

tEvents = crossingMovingAverages(close, 10, 50)

tEvents = tEvents.drop(['short_ma', 'long_ma', 'signal'], axis=1)

df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
df0 = df0[df0 > 0]
df0 = pd.Series(close.index[df0-1], index = close.index[close.shape[0] - df0.shape[0]:])
dailyReturns = close.loc[df0.index] / close.loc[df0.values].values - 1
threshold = dailyReturns.std()  
targets = dailyReturns.ewm(span = 100).std()

t1 = close.index.searchsorted(tEvents.index + pd.Timedelta(days=1))
t1 = t1[t1 < close.shape[0]]
t1 = pd.Series(close.index[t1], index = tEvents.index[:t1.shape[0]])
for i in tEvents.index:
    if i not in targets.index:
        targets.loc[i] = 0
targets = targets.loc[tEvents.index]
targets = targets[targets > 0.002]
ptsl = [1, 2] #Multipliers since side is given so can differentiate between horizontal barriers


events = pd.concat({'t1' : t1, 'trgts': targets, 'side': tEvents['positions']}, axis = 1).dropna(subset='trgts')

# This is the standard function of getting times of each barrier touched(as listed in Advances by Financial ML)
def applyPtSl(close, events, ptsl):
    out = events[['t1']].copy(deep = True)
    if ptsl[0]>0:pt=ptsl[0]*events['trgts']
    else:pt=pd.Series(index=events.index) 
    if ptsl[1]>0:sl=-ptsl[1]*events['trgts']
    else:sl=pd.Series(index=events.index)
    for loc,t1 in events['t1'].fillna(close.index[-1]).items():
        df0=close[loc:t1] 
        df0=(df0/close[loc]-1)*events.at[loc,'side'] 
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min()
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() 
    return out

df0 = applyPtSl(close=close, events = events, ptsl=ptsl)
events['t1'] = df0.dropna(how='all').min(axis=1) 

def getBins(events, close):
    events = events.dropna(subset = ['t1']) #I decided to add this to remove events where no barrier was touched for accuate labelling
    px = events.index.union(events['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    out = pd.DataFrame(index= events.index)
    out['ret'] = (px.loc[events['t1'].values].values / px.loc[events.index] - 1) * events['side']
    out['bin'] = 0
    
    
    out['bin'] = 1
    out.loc[out['ret']  <= 0, 'bin'] = 0
    
    return out

bins = getBins(events=events, close = close)


X = tEvents.drop('positions', axis =1).loc[bins.index]
y = bins['bin']

X = X.sort_index()
y = y.loc[X.index]

# I created a time based split since random would lead to info leakage
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

clf = RandomForestClassifier(n_estimators=100, max_depth = 5)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:,1]
high_conf_idx = proba > 0.75

high_confidence_accuracy = (y_test[high_conf_idx] == clf.predict(X_test)[high_conf_idx]).mean()

conf_events=  events.loc[X_test.index[high_conf_idx]]
conf_side = conf_events['side']


enter = close.loc[conf_events.index]
exit = close.loc[conf_events['t1']]


returns = (exit.values/enter.values - 1) * conf_side.values
print(returns)
hit_ratio = (returns > 0).mean()
avg_return = returns.mean()
sharpe = returns.mean() / returns.std()

print(f"Hit Ratio: {hit_ratio:.2f}")
print(f"Average Return per Trade: {avg_return:.5f}")
print(f"Sharpe Ratio: {sharpe:.2f}")


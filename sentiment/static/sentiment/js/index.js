// Load Plotly from CDN
(function loadPlotly(){
  if(window.Plotly) return;
  const s = document.createElement('script');
  s.src = 'https://cdn.plot.ly/plotly-latest.min.js';
  s.defer = true;
  document.head.appendChild(s);
})();

const form = document.getElementById('ticker-form');
const loading = document.getElementById('loading');
const content = document.getElementById('content');
const companyEl = document.getElementById('company');
const newsList = document.getElementById('news-list');
const rawSection = document.getElementById('raw');
const rawJson = document.getElementById('raw-json');
const rawToggle = document.getElementById('raw-toggle');
let lastUpdatedEl = null;
let inFlight = false;
let lastTopKeywords = null;

rawToggle.addEventListener('click', () => {
  rawSection.style.display = rawSection.style.display === 'none' ? 'block' : 'none';
});

function clearContent(){
  companyEl.innerHTML = '';
  newsList.innerHTML = '';
  rawJson.textContent = '';
}

function formatMarketCap(m){ return m ?? 'N/A'; }

function updateRiskGauge(riskScore, riskLevel, sentimentCounts, riskExplanation) {
  const score = riskScore || 50;
  const level = riskLevel || 'medium';

  document.getElementById('risk-score-display').textContent = Math.round(score);

  const circumference = 314;
  const percentage = score / 100;
  const dasharray = circumference * percentage;
  const arc = document.getElementById('risk-arc');
  arc.setAttribute('stroke-dasharray', `${dasharray} ${circumference}`);

  let color = '#f59e0b';
  if (level === 'low') {
    color = '#10b981';
  } else if (level === 'high') {
    color = '#ef4444';
  }
  arc.setAttribute('stroke', color);

  const badge = document.getElementById('risk-level-badge');
  badge.textContent = level.charAt(0).toUpperCase() + level.slice(1);
  badge.style.background = color;
  badge.style.color = '#fff';

  if (sentimentCounts) {
    document.getElementById('risk-positive-count').textContent = sentimentCounts.positive || 0;
    document.getElementById('risk-neutral-count').textContent = sentimentCounts.neutral || 0;
    document.getElementById('risk-negative-count').textContent = sentimentCounts.negative || 0;
  }

  const explanationEl = document.getElementById('risk-explanation');
  if (explanationEl) {
    if (riskExplanation && riskExplanation.summary) {
      const components = Array.isArray(riskExplanation.components) ? riskExplanation.components : [];
      const bulletText = components.map(c => `- ${c}`).join('\n');
      explanationEl.textContent = bulletText ? `${riskExplanation.summary}\n${bulletText}` : riskExplanation.summary;
      explanationEl.style.whiteSpace = 'pre-line';
    } else {
      explanationEl.textContent = 'Risk score is based on recent positive/neutral/negative news balance and model confidence.';
      explanationEl.style.whiteSpace = 'normal';
    }
  }
}


function renderTopDrivers(topKeywords) {
  const todayEl = document.getElementById('drivers-today');
  const weekEl = document.getElementById('drivers-week');
  const toggleBtn = document.getElementById('drivers-toggle');

  if (!todayEl || !weekEl) {
    return;
  }

  lastTopKeywords = topKeywords;
  const isExpanded = toggleBtn && toggleBtn.dataset.expanded === 'true';

  const todayList = (topKeywords && topKeywords.today) ? topKeywords.today : [];
  const weekList = (topKeywords && topKeywords.week) ? topKeywords.week : [];

  const renderList = (el, items) => {
    el.innerHTML = '';
    if (!items || items.length === 0) {
      el.innerHTML = '<span style="color:#9ca3af; font-size:0.75rem;">No recent drivers</span>';
      return;
    }
    const limit = isExpanded ? items.length : Math.min(items.length, 5);
    items.slice(0, limit).forEach(item => {
      const chip = document.createElement('span');
      chip.className = 'driver-chip';
      chip.textContent = `${item.keyword} (${item.count})`;
      el.appendChild(chip);
    });
  };

  renderList(todayEl, todayList);
  renderList(weekEl, weekList);

  if (toggleBtn) {
    const maxCount = Math.max(todayList.length, weekList.length);
    if (maxCount <= 5) {
      toggleBtn.style.display = 'none';
    } else {
      toggleBtn.style.display = 'inline-flex';
      toggleBtn.textContent = isExpanded ? 'Show less' : 'Show more';
      if (!toggleBtn.dataset.bound) {
        toggleBtn.addEventListener('click', () => {
          toggleBtn.dataset.expanded = toggleBtn.dataset.expanded === 'true' ? 'false' : 'true';
          if (lastTopKeywords) {
            renderTopDrivers(lastTopKeywords);
          }
        });
        toggleBtn.dataset.bound = 'true';
      }
    }
  }
}


function displayNewsList(newsArray) {
  if (newsArray.length === 0) {
    newsList.innerHTML = '<div style="color:#999; padding:20px; text-align:center;">No articles available.</div>';
    return;
  }

  newsList.innerHTML = '';
  newsArray.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'news-item';

    const titleEl = document.createElement('div');
    titleEl.style.cssText = 'margin-bottom:6px;';
    const a = document.createElement('a');
    a.href = item.link || '#';
    a.textContent = item.title || 'No title';
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.style.cssText = 'color:#2563eb; text-decoration:none; font-weight:500; font-size:0.95rem;';
    titleEl.appendChild(a);

    const metaRow = document.createElement('div');
    metaRow.style.cssText = 'display:flex; gap:8px; align-items:center; margin-top:6px;';
    const badge = sentimentBadge(item.sentiment_label);
    badge.style.padding = '3px 8px';
    badge.style.fontSize = '0.85rem';
    metaRow.appendChild(badge);

    if (item.sentiment_score != null) {
      const scoreEl = document.createElement('span');
      scoreEl.style.cssText = 'color:#666; font-size:0.85rem;';
      scoreEl.textContent = `Confidence: ${(item.sentiment_score * 100).toFixed(0)}%`;
      metaRow.appendChild(scoreEl);
    }

    const pubEl = document.createElement('div');
    pubEl.style.cssText = 'font-size:0.8rem; color:#999; margin-top:4px;';
    pubEl.textContent = item.published || 'Unknown date';

    div.appendChild(titleEl);
    div.appendChild(metaRow);
    div.appendChild(pubEl);
    newsList.appendChild(div);
  });
}

function renderTickerSuggestions(suggestions) {
  const containerFull = document.getElementById('suggestions-container-full');

  if (!suggestions || suggestions.length === 0) {
    containerFull.innerHTML = '<div style="color:#999; grid-column:1/-1; text-align:center; padding:20px;">No similar tickers found.</div>';
    return;
  }

  containerFull.innerHTML = '';

  suggestions.forEach(suggestion => {
    const ticker = suggestion.ticker;
    const sent = suggestion.sentiment;
    const price = suggestion.price;
    const change = suggestion.change;
    const changePct = suggestion.change_pct;

    let sentimentColor = '#f59e0b';
    let sentimentLabel = 'Neutral';
    let textColor = '#1f2937';
    if (sent.positive > sent.negative && sent.positive > sent.neutral) {
      sentimentColor = '#10b981';
      sentimentLabel = 'Positive';
      textColor = '#fff';
    } else if (sent.negative > sent.positive && sent.negative > sent.neutral) {
      sentimentColor = '#ef4444';
      sentimentLabel = 'Negative';
      textColor = '#fff';
    } else {
      textColor = '#1f2937';
    }

    const card = document.createElement('div');
    card.style.cssText = `
      padding: 14px;
      border: 2px solid ${sentimentColor};
      border-radius: 10px;
      background: ${sentimentColor}14;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: center;
    `;

    card.onmouseover = function() {
      this.style.transform = 'translateY(-4px)';
      this.style.boxShadow = `0 6px 16px ${sentimentColor}40`;
      this.style.borderColor = sentimentColor;
    };
    card.onmouseout = function() {
      this.style.transform = 'translateY(0)';
      this.style.boxShadow = 'none';
    };

    card.onclick = function() {
      document.getElementById('ticker').value = ticker;
      document.getElementById('ticker-form').dispatchEvent(new Event('submit'));
    };

    const tickerEl = document.createElement('div');
    tickerEl.style.cssText = `
      font-size: 1.4rem;
      font-weight: 700;
      color: #000;
      margin-bottom: 8px;
      letter-spacing: 0.5px;
    `;
    tickerEl.textContent = ticker;

    if (price != null) {
      const priceEl = document.createElement('div');
      priceEl.style.cssText = `
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 6px;
      `;
      priceEl.textContent = `$${price.toFixed(2)}`;

      if (change != null) {
        const changeEl = document.createElement('span');
        const isPositive = change > 0;
        const changeColor = isPositive ? '#10b981' : (change < 0 ? '#ef4444' : '#6b7280');
        const arrow = isPositive ? '▲' : (change < 0 ? '▼' : '—');
        changeEl.style.cssText = `
          font-size: 0.85rem;
          color: ${changeColor};
          font-weight: 600;
          margin-left: 6px;
        `;
        changeEl.textContent = `${arrow} ${Math.abs(change).toFixed(2)}`;
        if (changePct != null) {
          changeEl.textContent += ` (${changePct > 0 ? '+' : ''}${changePct.toFixed(2)}%)`;
        }
        priceEl.appendChild(changeEl);
      }

      card.appendChild(tickerEl);
      card.appendChild(priceEl);
    } else {
      card.appendChild(tickerEl);
    }

    const sentBadge = document.createElement('div');
    sentBadge.style.cssText = `
      display: inline-block;
      padding: 5px 10px;
      border-radius: 20px;
      background: ${sentimentColor};
      color: ${textColor};
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 10px;
    `;
    sentBadge.textContent = sentimentLabel;

    const statsEl = document.createElement('div');
    statsEl.style.cssText = `
      font-size: 0.8rem;
      color: #666;
      line-height: 1.6;
    `;
    statsEl.innerHTML = `
      <div style="margin: 2px 0;"><span style="color:#28a745; font-weight:600;">+${sent.positive}</span> | <span style="color:#ffc107; font-weight:600;">●${sent.neutral}</span> | <span style="color:#dc3545; font-weight:600;">−${sent.negative}</span></div>
      <div style="margin-top: 4px; font-size: 0.75rem; color: #999;">${sent.count} articles</div>
    `;

    card.appendChild(sentBadge);
    card.appendChild(statsEl);

    containerFull.appendChild(card);
  });
}

function sentimentBadge(label){
  const l = (label || '').toLowerCase();
  const span = document.createElement('span');
  span.className = 'badge ' + (l === 'positive' ? 'positive' : l === 'negative' ? 'negative' : 'neutral');
  span.textContent = label || 'Neutral';
  return span;
}

function resizePlot(plotId) {
  const el = document.getElementById(plotId);
  if (window.Plotly && el) {
    Plotly.Plots.resize(el);
  }
}

function resizeAllPlots() {
  resizePlot('sentiment-chart');
  resizePlot('candlestick-chart');
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const ticker = document.getElementById('ticker').value.trim();
  if(!ticker) return;

  clearContent();
  loading.style.display = 'block';
  content.style.display = 'none';

  try{
    const res = await fetch(`/analyze/?ticker=${encodeURIComponent(ticker)}`);
    if(!res.ok) throw new Error('Server returned ' + res.status);
    const data = await res.json();

    const c = data.company || {};
    const regime = data.regime || { regime: 'Unknown', color: '#999' };

    const title = document.createElement('div');
    title.innerHTML = `<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
      <h2 style="margin:0; font-size:1.8rem; font-weight:700;">${data.ticker}</h2>
      <div style="display:inline-block; padding:4px 10px; border-radius:16px; background:${regime.color}; color:white; font-size:0.75rem; font-weight:600; white-space:nowrap;">${regime.regime}</div>
    </div>
      <p style="margin:0 0 12px 0; font-size:1.1rem; opacity:0.9;">${c.name ?? 'Unknown Company'}</p>`;

    lastUpdatedEl = document.createElement('div');
    lastUpdatedEl.style.cssText = 'font-size:0.8rem; color:rgba(255,255,255,0.8); margin-top:8px;';
    lastUpdatedEl.textContent = '';

    const meta = document.createElement('div');
    meta.style.cssText = 'display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:16px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.2);';
    meta.className = 'company-meta';

    let priceHtml = 'N/A';
    let deltaHtml = '';
    if(c.current_price != null){
      priceHtml = `<div style="font-size:0.85rem; opacity:0.9; margin-bottom:4px;">Current Price</div>
        <div id="current-price" style="font-size:1.6rem; font-weight:700;">$${c.current_price.toFixed(2)}</div>`;
      if(c.change != null){
        const up = c.change > 0;
        const arrow = up ? '▲' : (c.change < 0 ? '▼' : '—');
        const color = up ? '#4caf50' : (c.change < 0 ? '#f44336' : '#999');
        deltaHtml = `<div id="current-delta" style="color:${color}; font-weight:600; margin-top:4px; font-size:0.9rem;"> ${arrow} ${Math.abs(c.change).toFixed(2)} ${c.change_pct!=null? '('+c.change_pct.toFixed(2)+'%)':''}</div>`;
      } else {
        deltaHtml = `<div id="current-delta"></div>`;
      }
    } else {
      priceHtml = '<div style="font-size:0.85rem; opacity:0.9; margin-bottom:4px;">Current Price</div><div id="current-price" style="font-size:1.4rem; font-weight:700;">N/A</div>';
      deltaHtml = `<div id="current-delta"></div>`;
    }
    meta.innerHTML = `
      <div>${priceHtml}${deltaHtml}</div>
      <div>
        <div style="font-size:0.85rem; opacity:0.9; margin-bottom:4px;">Sector</div>
        <div style="font-size:1rem; font-weight:600;">${c.sector ?? 'N/A'}</div>
      </div>
      <div>
        <div style="font-size:0.85rem; opacity:0.9; margin-bottom:4px;">Market Cap</div>
        <div style="font-size:1rem; font-weight:600;">${formatMarketCap(c.market_cap)}</div>
      </div>
    `;

    const summary = document.createElement('p');
    summary.style.cssText = 'margin:12px 0 0 0; font-size:0.95rem; opacity:0.95; line-height:1.5;';
    summary.textContent = c.business_summary ?? '';

    companyEl.appendChild(title);
    companyEl.appendChild(meta);
    companyEl.appendChild(summary);
    companyEl.appendChild(lastUpdatedEl);

    const news = data.news || [];
    window.allNewsData = news;

    if(news.length === 0){
      newsList.innerHTML = '<div>No recent news or sentiment analysis unavailable.</div>';
    } else {
      displayNewsList(news);
    }

    try{
      const counts = { Positive:0, Negative:0, Neutral:0 };
      news.forEach(n => {
        const label = n.sentiment_label || (n.label || 'Neutral');
        if(label.toLowerCase().startsWith('posit')) counts.Positive++;
        else if(label.toLowerCase().startsWith('neg')) counts.Negative++;
        else counts.Neutral++;
      });

      const labels = [];
      const values = [];
      const colors = [];
      if(counts.Positive>0){ labels.push('Positive'); values.push(counts.Positive); colors.push('#28a745'); }
      if(counts.Negative>0){ labels.push('Negative'); values.push(counts.Negative); colors.push('#dc3545'); }
      if(counts.Neutral>0){ labels.push('Neutral'); values.push(counts.Neutral); colors.push('#6c757d'); }

      const chartDiv = document.getElementById('sentiment-chart');
      if(window.Plotly && labels.length>0){
        Plotly.newPlot(chartDiv, [{ values, labels, type:'pie', marker:{colors}, hole:0.4 }], {height:300, margin:{t:20,b:20,l:20,r:20}});
      } else {
        chartDiv.textContent = labels.length ? `${labels.map((l,i)=>`${l}: ${values[i]}`).join('\n')}` : 'No sentiment data';
      }
    }catch(e){
      console.error('Chart error', e);
    }

    rawJson.textContent = JSON.stringify(data, null, 2);

    updateRiskGauge(data.risk_score, data.risk_level, data.risk_sentiment_counts, data.risk_explanation);

    renderTopDrivers(data.top_keywords || {});

    const suggestions = data.ticker_suggestions || [];
    renderTickerSuggestions(suggestions);

    try{
      const hist = data.history || [];
      const chartDiv = document.getElementById('candlestick-chart');
      if(hist.length > 0 && window.Plotly){
        const x = hist.map(h => h.date);
        const open = hist.map(h => h.open);
        const high = hist.map(h => h.high);
        const low = hist.map(h => h.low);
        const close = hist.map(h => h.close);

        const trace = { x, open, high, low, close, type: 'candlestick', increasing: {line: {color: '#28a745'}}, decreasing: {line: {color: '#dc3545'}} };
        const layout = { margin:{t:20,b:30,l:40,r:20}, xaxis:{rangeslider:{visible:false}}, height:420 };
        Plotly.newPlot(chartDiv, [trace], layout, {responsive:true});
      } else {
        document.getElementById('candlestick-chart').textContent = hist.length ? 'Plotly unavailable' : 'No historical data';
      }
    }catch(e){
      console.error('Candlestick chart error', e);
    }

    content.style.display = 'flex';
    content.style.flexDirection = 'column';
    loading.style.display = 'none';
    requestAnimationFrame(() => resizeAllPlots());
    if(lastUpdatedEl){
      lastUpdatedEl.textContent = 'Last updated: ' + new Date().toLocaleString();
    }
  }catch(err){
    loading.textContent = 'Error: ' + (err.message || err);
  }
});

window.addEventListener('load', () => form.dispatchEvent(new Event('submit')));

async function pollPrice(){
  if(document.getElementById('content').style.display === 'none' || inFlight) return;
  const ticker = document.getElementById('ticker').value.trim();
  if(!ticker) return;

  inFlight = true;
  try{
    const res = await fetch(`/price/?ticker=${encodeURIComponent(ticker)}`);
    if(!res.ok) throw new Error('Server returned ' + res.status);
    const p = await res.json();
    const priceEl = document.getElementById('current-price');
    const deltaEl = document.getElementById('current-delta');
    if(priceEl && p.current_price != null){
      priceEl.textContent = '$' + p.current_price.toFixed(2);
    }
    if(deltaEl){
      if(p.change != null){
        const up = p.change > 0;
        const arrow = up ? '▲' : (p.change < 0 ? '▼' : '');
        const color = up ? 'green' : (p.change < 0 ? 'red' : '#666');
        deltaEl.style.color = color;
        deltaEl.style.fontWeight = '700';
        deltaEl.textContent = ` ${arrow} ${Math.abs(p.change).toFixed(2)} ${p.change_pct!=null? '('+p.change_pct.toFixed(2)+'% )':''}`;
      } else {
        deltaEl.textContent = '';
      }
    }

    if(p.risk_score != null){
      updateRiskGauge(p.risk_score, p.risk_level, p.risk_sentiment_counts, p.risk_explanation);
    }

    if(lastUpdatedEl){ lastUpdatedEl.textContent = 'Last updated: ' + new Date().toLocaleString(); }
  }catch(e){
    console.warn('Price poll failed', e);
  }finally{ inFlight = false; }
}

setInterval(pollPrice, 30000);

window.addEventListener('resize', () => {
  if (content.style.display !== 'none') {
    resizeAllPlots();
  }
});

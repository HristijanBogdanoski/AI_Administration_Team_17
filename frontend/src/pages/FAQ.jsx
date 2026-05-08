import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Footer from '../components/Footer';

// ─── Floating Chat Widget ──────────────────────────────────────────────
function ChatWidget({ onExpand }) {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState([
    { type: 'bot', text: 'Здраво! Јас сум АИ Асистентот на е-Влада. Како можам да ви помогнам денес?' }
  ]);
  const [input, setInput] = useState('');
  const msgsRef = useRef(null);

  useEffect(() => {
    if (msgsRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight;
  }, [msgs]);

  const getTime = () => new Date().toLocaleTimeString('mk', { hour: '2-digit', minute: '2-digit' });

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    setMsgs(p => [...p, { type: 'user', text }]);
    setInput('');
    try {
      const r = await fetch('http://127.0.0.1:8000/faq/search', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, top_k: 1 })
      });
      const d = await r.json();
      if (r.ok && d.results?.length) {
        setMsgs(p => [...p, { type: 'bot', text: d.results[0].answer }]);
      } else {
        setMsgs(p => [...p, { type: 'bot', text: 'За подетални информации посетете gov.mk.' }]);
      }
    } catch {
      setMsgs(p => [...p, { type: 'bot', text: 'Грешка при поврзување. Обидете се повторно.' }]);
    }
  };

  return (
    <div style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 1000, fontFamily: "'Sora', sans-serif" }}>
      {open && (
        <div style={{
          position: 'absolute', bottom: 72, right: 0, width: 320,
          background: '#fff', borderRadius: 16, boxShadow: '0 8px 32px rgba(0,0,0,0.18)',
          overflow: 'hidden', display: 'flex', flexDirection: 'column'
        }}>
          {/* Header */}
          <div style={{ background: '#1B3A6B', padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 36, height: 36, background: '#D4A017', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16, flexShrink: 0 }}>🤖</div>
            <div style={{ flex: 1 }}>
              <div style={{ color: '#fff', fontWeight: 700, fontSize: '0.88rem' }}>АИ Асистент</div>
              <div style={{ color: '#93c5fd', fontSize: '0.72rem' }}>е-Влада Поддршка</div>
            </div>
            <button
              onClick={() => { setOpen(false); onExpand(); }}
              style={{ background: 'none', border: 'none', color: '#93c5fd', fontSize: '0.78rem', cursor: 'pointer', fontFamily: "'Sora', sans-serif", fontWeight: 600 }}
            >Прошири</button>
            <button
              onClick={() => setOpen(false)}
              style={{ background: 'none', border: 'none', color: '#93c5fd', fontSize: '1.2rem', cursor: 'pointer', lineHeight: 1 }}
            >×</button>
          </div>

          {/* Messages */}
          <div ref={msgsRef} style={{ flex: 1, maxHeight: 280, overflowY: 'auto', padding: 14, display: 'flex', flexDirection: 'column', gap: 10, background: '#f4f6fa' }}>
            {msgs.map((m, i) => (
              <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-end', flexDirection: m.type === 'user' ? 'row-reverse' : 'row' }}>
                {m.type === 'bot' && (
                  <div style={{ width: 28, height: 28, background: '#D4A017', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, flexShrink: 0 }}>🤖</div>
                )}
                <div>
                  <div style={{
                    padding: '9px 12px', borderRadius: 12, fontSize: '0.83rem', lineHeight: 1.6, maxWidth: 200,
                    background: m.type === 'bot' ? '#fff' : '#1B3A6B',
                    color: m.type === 'bot' ? '#1e293b' : '#fff',
                    borderBottomLeftRadius: m.type === 'bot' ? 3 : 12,
                    borderBottomRightRadius: m.type === 'user' ? 3 : 12,
                    boxShadow: '0 1px 4px rgba(0,0,0,0.06)'
                  }}>{m.text}</div>
                  <div style={{ fontSize: '0.65rem', color: '#94a3b8', marginTop: 3, textAlign: m.type === 'user' ? 'right' : 'left' }}>{getTime()}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Input */}
          <div style={{ padding: 10, background: '#fff', borderTop: '1px solid #e2e8f0', display: 'flex', gap: 8 }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="Напишете прашање..."
              style={{ flex: 1, padding: '9px 12px', border: '1.5px solid #e2e8f0', borderRadius: 10, fontSize: '0.82rem', fontFamily: "'Sora', sans-serif", outline: 'none', background: '#f8fafc' }}
            />
            <button
              onClick={send}
              style={{ width: 38, height: 38, background: '#1B3A6B', border: 'none', borderRadius: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 14, color: '#fff' }}
            >➤</button>
          </div>
        </div>
      )}

      {/* FAB */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: 56, height: 56, background: '#D4A017', border: 'none', borderRadius: '50%',
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 22, boxShadow: '0 4px 16px rgba(212,160,23,0.45)',
          transition: 'all 0.2s', position: 'relative'
        }}
      >
        💬
        <span style={{
          position: 'absolute', top: -2, right: -2, width: 18, height: 18,
          background: '#1B3A6B', borderRadius: '50%', border: '2px solid #D4A017',
          color: '#fff', fontSize: '0.6rem', fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>АИ</span>
      </button>
    </div>
  );
}

// ─── FAQ Accordion Item ────────────────────────────────────────────────
function FaqItem({ q, a }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      background: '#fff', borderRadius: 12, border: '1.5px solid #e2e8f0',
      overflow: 'hidden', transition: 'box-shadow 0.2s',
      boxShadow: open ? '0 4px 16px rgba(15,32,68,0.08)' : 'none'
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', padding: '18px 20px', background: 'none', border: 'none',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          cursor: 'pointer', fontFamily: "'Sora', sans-serif", textAlign: 'left', gap: 12
        }}
      >
        <span style={{ fontWeight: 600, fontSize: '0.92rem', color: '#1e293b', lineHeight: 1.4 }}>{q}</span>
        <span style={{
          width: 28, height: 28, background: open ? '#1B3A6B' : '#EFF6FF',
          borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1rem', color: open ? '#fff' : '#1B3A6B', flexShrink: 0,
          transition: 'all 0.2s', transform: open ? 'rotate(45deg)' : 'none'
        }}>+</span>
      </button>
      {open && (
        <div style={{ padding: '0 20px 18px', color: '#475569', fontSize: '0.87rem', lineHeight: 1.75, borderTop: '1px solid #f0f4ff' }}>
          {a}
        </div>
      )}
    </div>
  );
}

// ─── FAQ DATA ──────────────────────────────────────────────────────────
const FAQ_DATA = {
  Документи: [
    { q: 'Кои документи се потребни за пасош?', a: 'За добивање пасош потребно е: лична карта, уплатница за такса (1.100 ден.), 1 биометриска фотографија и пополнето барање. Поднесете го барањето во МВР. Рок на издавање: 30 дена, итна постапка: 5 работни дена.' },
    { q: 'Како да добијам лична карта?', a: 'Потребни документи: извод од матична книга на родените, уплатница (300 ден.) и претходната лична карта. Рок на издавање: 15 работни дена. Поднесете барање во МВР.' },
    { q: 'Постапка за возачка дозвола?', a: 'Потребно е: важечка лична карта, лекарско уверение (не постаро од 6 месеци), положен теоретски и практичен испит. Таксата изнесува 1.200 ден.' },
    { q: 'Колку важи пасошот?', a: 'Пасошот за возрасни важи 10 години, а за деца до 18 години – 5 години.' },
  ],
  Даноци: [
    { q: 'Кој е рокот за даночна пријава?', a: 'Годишната даночна пријава за физички лица се поднесува до 15 март. Пријавата може да се поднесе електронски преку порталот на УЈП (ujp.gov.mk) или лично на шалтер.' },
    { q: 'Како се плаќа данокот онлајн?', a: 'Данокот може да се плати онлајн преку e-плаќање на порталот na.mk или преку интернет банкарство со уплатница. Исто така може да се плати во банка или пошта.' },
    { q: 'Каде да пријавам промена на адреса за даноци?', a: 'Промената на адреса ја пријавувате во УЈП – Управа за јавни приходи, лично или преку нивниот онлајн систем на ujp.gov.mk.' },
  ],
  Социјални: [
    { q: 'Кои се условите за социјална помош?', a: 'Право на социјална помош имаат граѓани чии месечни приходи не ги покриваат основните потреби. Барањето се поднесува во општинскиот Центар за социјална работа со докази за приходи и имотна состојба.' },
    { q: 'Каде да поднесам барање за пензија?', a: 'Барањето за пензија се поднесува во Фондот за пензиско и инвалидско осигурување (ПИОМ). Потребни документи: лична карта, работна книшка и потврди за стаж.' },
    { q: 'Дали постои детски додаток?', a: 'Да, детскиот додаток се остварува преку Центарот за социјална работа. Право имаат родители/старатели на деца до 18 години, врз основа на имотна состојба.' },
  ],
  'Онлајн Услуги': [
    { q: 'Кои услуги се достапни онлајн?', a: 'На порталот е-Влада достапни се над 150 онлајн услуги: поднесување барања за документи, плаќање такси, проверка на статус, закажување термини, пристап до регистри и многу повеќе.' },
    { q: 'Дали треба сметка за е-Влада?', a: 'За повеќето услуги потребна е регистрирана сметка. Регистрацијата е бесплатна и се прави на порталот. Некои информативни услуги се достапни без регистрација.' },
  ],
  Плаќање: [
    { q: 'Kako да платам такса онлајн?', a: 'Таксите се плаќаат преку порталот na.mk со банкарска картичка или интернет банкарство. По плаќањето добивате потврда на е-маил.' },
    { q: 'Дали можам да платам на шалтер?', a: 'Да, сите такси и придонеси може да се платат на шалтерите на банките, поштите или во самата институција.' },
  ],
};

const CATEGORIES = ['Сите', ...Object.keys(FAQ_DATA)];



// ─── MAIN FAQ PAGE ─────────────────────────────────────────────────────
export default function FAQ() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState('Сите');
  const [extraItems, setExtraItems] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newQ, setNewQ] = useState('');
  const [newA, setNewA] = useState('');
  const [newCat, setNewCat] = useState(Object.keys(FAQ_DATA)[0]);

  // Decode JWT to get role
  const userRole = (() => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return null;
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.role || null;
    } catch {
      return null;
    }
  })();

  const handleAddFaq = () => {
    if (!newQ.trim() || !newA.trim()) return;
    setExtraItems(prev => [...prev, { q: newQ.trim(), a: newA.trim(), cat: newCat }]);
    setNewQ(''); setNewA(''); setNewCat(Object.keys(FAQ_DATA)[0]);
    setShowAddModal(false);
  };

  // Build filtered list
  const allItems = [...Object.entries(FAQ_DATA).flatMap(([cat, items]) => items.map(i => ({ ...i, cat }))), ...extraItems];
  const filtered = allItems.filter(item => {
    const matchCat = activeCategory === 'Сите' || item.cat === activeCategory;
    const matchSearch = !search || item.q.toLowerCase().includes(search.toLowerCase()) || item.a.toLowerCase().includes(search.toLowerCase());
    return matchCat && matchSearch;
  });

  // Group by category for display
  const grouped = {};
  filtered.forEach(item => {
    if (!grouped[item.cat]) grouped[item.cat] = [];
    grouped[item.cat].push(item);
  });

  const s = { fontFamily: "'Sora', sans-serif" };

  return (
    <div style={{ background: '#F4F6FA', minHeight: '100vh', display: 'flex', flexDirection: 'column', ...s }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&display=swap');`}</style>

      {/* HERO */}
      <div style={{
        background: 'linear-gradient(150deg, #0f2044 0%, #1B3A6B 55%, #1a4a8a 100%)',
        padding: '52px 40px 68px', textAlign: 'center', position: 'relative', overflow: 'hidden'
      }}>
        <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 60% 70% at 50% 120%, rgba(212,160,23,0.1) 0%, transparent 70%)' }} />
        <div style={{
          width: 64, height: 64, background: 'rgba(255,255,255,0.1)', backdropFilter: 'blur(12px)',
          borderRadius: 18, display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 22px', border: '1px solid rgba(255,255,255,0.15)', position: 'relative', zIndex: 2, fontSize: 28
        }}>❓</div>
        <h1 style={{ color: '#fff', fontSize: '2.2rem', fontWeight: 800, letterSpacing: '-0.02em', margin: '0 0 10px', position: 'relative', zIndex: 2 }}>
          Често Поставувани Прашања
        </h1>
        <p style={{ color: '#93c5fd', fontSize: '0.92rem', margin: 0, position: 'relative', zIndex: 2 }}>
          Пронајдете брзи одговори на најчестите прашања
        </p>
      </div>

      {/* SEARCH + FILTERS */}
      <div style={{ background: '#fff', borderBottom: '1px solid #e2e8f0', padding: '28px 40px', position: 'sticky', top: 64, zIndex: 50 }}>
        <div style={{ maxWidth: 760, margin: '0 auto' }}>
          {/* Search */}
          <div style={{ position: 'relative', marginBottom: 20 }}>
            <span style={{ position: 'absolute', left: 16, top: '50%', transform: 'translateY(-50%)', color: '#94a3b8', fontSize: 16 }}>🔍</span>
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Пребарај прашања..."
              style={{
                width: '100%', padding: '14px 16px 14px 44px', border: '1.5px solid #e2e8f0',
                borderRadius: 12, fontSize: '0.92rem', fontFamily: "'Sora', sans-serif",
                outline: 'none', background: '#f8fafc', color: '#1e293b', boxSizing: 'border-box',
                transition: 'border 0.2s'
              }}
              onFocus={e => e.target.style.borderColor = '#1B3A6B'}
              onBlur={e => e.target.style.borderColor = '#e2e8f0'}
            />
          </div>

          {/* Category chips + Add button */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                style={{
                  padding: '8px 18px', borderRadius: 24, border: '1.5px solid',
                  borderColor: activeCategory === cat ? '#1B3A6B' : '#e2e8f0',
                  background: activeCategory === cat ? '#1B3A6B' : 'transparent',
                  color: activeCategory === cat ? '#fff' : '#475569',
                  fontSize: '0.82rem', fontWeight: activeCategory === cat ? 700 : 500,
                  cursor: 'pointer', fontFamily: "'Sora', sans-serif", transition: 'all 0.2s',
                  display: 'flex', alignItems: 'center', gap: 6
                }}
              >
                {cat !== 'Сите' && <span style={{ fontSize: 12 }}>🏷</span>}
                {cat}
              </button>
            ))}
            </div>
            {userRole === 'admin' && (
              <button
                onClick={() => setShowAddModal(true)}
                style={{ background: '#1B3A6B', border: 'none', color: '#fff', padding: '8px 16px', borderRadius: 24, fontSize: '0.82rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}
              >
                <span style={{ fontSize: 16 }}>+</span> Додади прашање
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ADD FAQ MODAL */}
      {showAddModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onClick={() => setShowAddModal(false)}>
          <div style={{ background: '#fff', borderRadius: 16, padding: 32, width: '100%', maxWidth: 500, position: 'relative' }}
            onClick={e => e.stopPropagation()}>
            <h3 style={{ margin: '0 0 20px', color: '#1e293b', fontSize: '1.1rem', fontWeight: 700 }}>Додади ново прашање</h3>

            <label style={{ display: 'block', fontSize: '0.82rem', color: '#475569', marginBottom: 6 }}>Категорија</label>
            <select value={newCat} onChange={e => setNewCat(e.target.value)} style={{ width: '100%', padding: '10px 12px', borderRadius: 8, border: '1.5px solid #e2e8f0', fontSize: '0.88rem', marginBottom: 16, outline: 'none', background: '#f8fafc' }}>
              {Object.keys(FAQ_DATA).map(c => <option key={c} value={c}>{c}</option>)}
            </select>

            <label style={{ display: 'block', fontSize: '0.82rem', color: '#475569', marginBottom: 6 }}>Прашање</label>
            <input value={newQ} onChange={e => setNewQ(e.target.value)} placeholder="Внесете прашање..." style={{ width: '100%', padding: '10px 12px', borderRadius: 8, border: '1.5px solid #e2e8f0', fontSize: '0.88rem', marginBottom: 16, outline: 'none', background: '#f8fafc', boxSizing: 'border-box' }} />

            <label style={{ display: 'block', fontSize: '0.82rem', color: '#475569', marginBottom: 6 }}>Одговор</label>
            <textarea value={newA} onChange={e => setNewA(e.target.value)} placeholder="Внесете одговор..." rows={4} style={{ width: '100%', padding: '10px 12px', borderRadius: 8, border: '1.5px solid #e2e8f0', fontSize: '0.88rem', marginBottom: 20, outline: 'none', background: '#f8fafc', boxSizing: 'border-box', resize: 'vertical', fontFamily: 'inherit' }} />

            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button onClick={() => setShowAddModal(false)} style={{ padding: '9px 20px', borderRadius: 8, border: '1.5px solid #e2e8f0', background: 'transparent', color: '#475569', cursor: 'pointer', fontSize: '0.88rem' }}>Откажи</button>
              <button onClick={handleAddFaq} style={{ padding: '9px 20px', borderRadius: 8, border: 'none', background: '#1B3A6B', color: '#fff', cursor: 'pointer', fontSize: '0.88rem', fontWeight: 700 }}>Зачувај</button>
            </div>
          </div>
        </div>
      )}

      {/* FAQ CONTENT */}
      <div style={{ maxWidth: 760, margin: '40px auto 60px', padding: '0 20px', flex: 1, width: '100%' }}>
        {Object.keys(grouped).length === 0 ? (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: '#94a3b8' }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🔍</div>
            <p style={{ fontSize: '1rem', fontWeight: 600 }}>Нема резултати за „{search}"</p>
            <p style={{ fontSize: '0.85rem', marginTop: 8 }}>Обидете се со поинакви зборови.</p>
          </div>
        ) : (
          Object.entries(grouped).map(([cat, items]) => (
            <div key={cat} style={{ marginBottom: 36 }}>
              {activeCategory === 'Сите' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                  <span style={{ fontSize: '0.7rem', fontWeight: 700, color: '#94a3b8', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{cat}</span>
                  <div style={{ flex: 1, height: 1, background: '#e2e8f0' }} />
                  <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>{items.length} прашања</span>
                </div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {items.map((item, i) => <FaqItem key={i} q={item.q} a={item.a} />)}
              </div>
            </div>
          ))
        )}

        {/* CTA BANNER */}
        <div style={{
          background: '#1B3A6B', borderRadius: 16, padding: '24px 32px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          flexWrap: 'wrap', gap: 16, marginTop: 8
        }}>
          <div>
            <h3 style={{ color: '#fff', fontWeight: 700, fontSize: '1rem', marginBottom: 6 }}>Не го пронајдовте одговорот?</h3>
            <p style={{ color: '#93c5fd', fontSize: '0.85rem', margin: 0 }}>Нашиот АИ асистент е тука да помогне</p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={() => navigate('/chat')}
              style={{ background: '#D4A017', color: '#0f2044', border: 'none', padding: '11px 24px', borderRadius: 10, fontWeight: 700, cursor: 'pointer', fontSize: '0.88rem', fontFamily: "'Sora', sans-serif" }}
            >АИ Чет</button>
            <button style={{ background: 'transparent', color: '#fff', border: '1.5px solid rgba(255,255,255,0.3)', padding: '11px 24px', borderRadius: 10, fontWeight: 600, cursor: 'pointer', fontSize: '0.88rem', fontFamily: "'Sora', sans-serif" }}>
              Јавете се
            </button>
          </div>
        </div>
      </div>

      <Footer />

      {/* Floating chat widget */}
      <ChatWidget onExpand={() => navigate('/chat')} />
    </div>
  );
}
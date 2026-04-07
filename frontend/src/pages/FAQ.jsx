import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState(null);
  const [searchTerm, setSearchTerm] = useState(""); // Состојба за пребарување

  const brandBlue = '#2e3e77';
  const brandGold = '#FFB800';
  const lightBlue = '#F0F4F8';

  const faqData = [
    { cat: "Документи", q: "Како да поднесам барање за пасош?", a: "За пасош можете да поднесете барање лично на најблискиот шалтер на МВР или онлајн преку порталот е-Влада. Потребни документи: лична карта или акт за раѓање, фотографија, доказ за уплата на такса (2.800 МКД). Рокот за обработка е 15 работни дена." },
    { cat: "Онлајн Услуги", q: "Кои услуги се достапни онлајн?", a: "Достапни се над 150 услуги вклучувајќи изводи, даночни пријави и социјална помош." },
    { cat: "Документи", q: "Колку трае издавањето на лична карта?", a: "Стандардниот рок е 15 дена, но постои и брза постапка." },
    { cat: "Даноци", q: "Кога е рокот за поднесување даночна пријава?", a: "Рокот за годишна даночна пријава обично е до 15-ти март." },
    { cat: "Плаќање", q: "Кои начини на плаќање се прифаќаат?", a: "Прифаќаме сите дебитни и кредитни картички, како и е-банкарство." },
  ];

  // ЛОГИКА ЗА ФИЛТРИРАЊЕ: Проверува дали текстот го има во прашањето или во категоријата
  const filteredData = faqData.filter((item) =>
    item.q.toLowerCase().includes(searchTerm.toLowerCase()) || 
    item.cat.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div style={{ backgroundColor: '#fff', minHeight: '100vh', fontFamily: 'Arial, sans-serif' }}>
      
      {/* 1. BLUE HERO SECTION */}
      <div style={{ background: brandBlue, color: 'white', padding: '60px 20px', textAlign: 'center' }}>
        <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '10px', color: '#ffffff' }}>Често Поставувани Прашања</h1>
        <p style={{ opacity: 0.8, fontSize: '14px', color: '#ffffff' }}>Пронајдете брзи одговори на најчестите прашања</p>
      </div>

      <div style={{ maxWidth: '900px', margin: '-30px auto 60px auto', padding: '0 20px' }}>
        
        {/* 2. SEARCH BAR (Функционален) */}
        <div style={{ position: 'relative', marginBottom: '30px' }}>
          <input 
            type="text" 
            placeholder="Пребарај прашања..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ 
              width: '100%', padding: '15px 45px', borderRadius: '30px', border: '1px solid #ddd', 
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)', fontSize: '14px', outline: 'none'
            }} 
          />
          <span style={{ position: 'absolute', left: '20px', top: '15px', opacity: 0.4 }}>🔍</span>
        </div>

        {/* 3. FILTERS / TAGS */}
        <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap', marginBottom: '40px' }}>
          {['Сите', 'Документи', 'Даноци', 'Социјални', 'Онлајн Услуги', 'Плаќање'].map((tag, i) => (
            <button 
              key={i} 
              onClick={() => tag === 'Сите' ? setSearchTerm("") : setSearchTerm(tag)}
              style={{
                padding: '8px 18px', borderRadius: '20px', border: '1px solid #eee',
                background: (searchTerm === tag || (tag === 'Сите' && searchTerm === "")) ? brandBlue : 'white',
                color: (searchTerm === tag || (tag === 'Сите' && searchTerm === "")) ? 'white' : '#666',
                fontSize: '13px', cursor: 'pointer', fontWeight: '500', transition: '0.2s'
            }}>{tag}</button>
          ))}
        </div>

        {/* 4. FAQ LIST */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {filteredData.length > 0 ? (
            filteredData.map((item, index) => (
              <div key={index} style={{ 
                border: openIndex === index ? `1px solid ${brandBlue}` : '1px solid #eee',
                borderRadius: '8px', overflow: 'hidden', background: 'white'
              }}>
                <div 
                  onClick={() => setOpenIndex(openIndex === index ? null : index)}
                  style={{ padding: '15px 20px', display: 'flex', alignItems: 'center', cursor: 'pointer', background: openIndex === index ? '#f8f9fa' : 'white' }}
                >
                  <span style={{ 
                    fontSize: '11px', fontWeight: 'bold', color: brandBlue, background: lightBlue, 
                    padding: '4px 10px', borderRadius: '4px', marginRight: '15px', minWidth: '100px', textAlign: 'center'
                  }}>
                    {item.cat}
                  </span>
                  <span style={{ flex: 1, fontWeight: '600', fontSize: '15px', color: '#333' }}>{item.q}</span>
                  <span style={{ color: '#ccc', transform: openIndex === index ? 'rotate(180deg)' : 'rotate(0)', transition: '0.3s' }}>▼</span>
                </div>
                
                {openIndex === index && (
                  <div style={{ padding: '20px', fontSize: '14px', color: '#666', borderTop: '1px solid #eee', lineHeight: '1.6', background: '#fff' }}>
                    {item.a}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
              Нема резултати за "{searchTerm}"
            </div>
          )}
        </div>

        {/* 5. BOTTOM CTA BANNER */}
        <div style={{ 
          background: brandBlue, borderRadius: '12px', padding: '30px', marginTop: '50px',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'white'
        }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '18px', color: 'white' }}>Не го пронајдовте одговорот?</h3>
            <p style={{ margin: '5px 0 0 0', opacity: 0.8, fontSize: '13px', color: 'white' }}>Нашиот АИ асистент е тука да помогне</p>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <Link 
              to="/ai" 
              style={{ 
                background: brandGold, color: brandBlue, padding: '10px 20px', 
                borderRadius: '6px', fontWeight: 'bold', textDecoration: 'none', fontSize: '14px' 
              }}
            >
              АИ Чет
            </Link>
            <button style={{ background: 'transparent', border: '1px solid white', padding: '10px 20px', borderRadius: '6px', color: 'white', cursor: 'pointer' }}>
                Јавете се
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
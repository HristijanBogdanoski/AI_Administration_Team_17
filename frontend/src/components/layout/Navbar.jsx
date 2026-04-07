import React from 'react';
import { Link } from 'react-router-dom';

export default function Navbar() {
  const brandBlue = '#2e3e77';
  const brandGold = '#FFB800';

  const navStyle = {
    background: brandBlue,
    padding: '15px 0',
    width: '100%',
    boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
    position: 'sticky',
    top: '0',
    zIndex: '1000'
  };

  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  };

  // Стил за линковите во менито
  const navLinkStyle = {
    color: 'white',
    textDecoration: 'none',
    margin: '0 15px',
    fontSize: '15px',
    fontWeight: '500',
    transition: 'color 0.3s ease'
  };

  const buttonStyle = {
    background: brandGold,
    color: brandBlue,
    border: 'none',
    padding: '8px 20px',
    borderRadius: '4px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'transform 0.2s ease'
  };

  // Функција за hover ефект на линковите
  const handleMouseEnter = (e) => {
    e.target.style.color = brandGold;
  };

  const handleMouseLeave = (e) => {
    e.target.style.color = 'white';
  };

  return (
    <nav style={navStyle}>
      <div style={containerStyle}>
        
        {/* ЛЕВО: ЛОГО (Носи на почетна) */}
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '10px', textDecoration: 'none', flex: 1 }}>
          <div style={{ 
            background: brandGold, 
            width: '32px', 
            height: '32px', 
            borderRadius: '4px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            fontWeight: 'bold', 
            color: brandBlue 
          }}>е</div>
          <div style={{ color: 'white', lineHeight: '1.1' }}>
            <div style={{ fontWeight: 'bold', fontSize: '18px' }}>е-Влада</div>
            <div style={{ fontSize: '10px', opacity: 0.7 }}>Јавни Услуги</div>
          </div>
        </Link>

        {/* СРЕДИНА: МЕНИ (Сега со точни патеки) */}
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 2 }}>
          <Link to="/" style={navLinkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>Дома</Link>
          <Link to="/faq" style={navLinkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>ЧПП</Link>
          <Link to="/services" style={navLinkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>Услуги</Link>
          <Link to="/locations" style={navLinkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>Локација</Link>
          
          {/* КЛУЧНАТА ПОПРАВКА: Овој линк сега мора да води до /ai */}
          <Link to="/ai" style={navLinkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>АИ Чат</Link>
        </div>

        {/* ДЕСНО: КОПЧЕ НАЈАВА */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', flex: 1 }}>
          <button 
            style={buttonStyle}
            onMouseOver={(e) => e.target.style.transform = 'scale(1.05)'}
            onMouseOut={(e) => e.target.style.transform = 'scale(1)'}
          >
            Најава
          </button>
        </div>

      </div>
    </nav>
  );
}
import React from 'react';
import { Link } from 'react-router-dom'; // Важно за брза навигација

export default function Footer() {
  const brandBlue = '#2e3e77'; 
  const brandGold = '#FFB800';

  const footerStyle = {
    background: brandBlue,
    color: '#ffffff',
    padding: '60px 0 0 0',
    marginTop: 'auto',
    width: '100%',
    fontFamily: 'Arial, sans-serif'
  };

  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 20px 60px 20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
    gap: '40px'
  };

  const columnStyle = {
    flex: '1',
    minWidth: '250px',
    display: 'flex',
    flexDirection: 'column'
  };

  const linkStyle = {
    color: '#ffffff',
    textDecoration: 'none',
    opacity: 0.7,
    fontSize: '15px',
    marginBottom: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    transition: 'all 0.3s ease'
  };

  const handleMouseEnter = (e) => {
    e.target.style.opacity = '1';
    e.target.style.paddingLeft = '5px';
  };

  const handleMouseLeave = (e) => {
    e.target.style.opacity = '0.7';
    e.target.style.paddingLeft = '0px';
  };

  return (
    <footer style={footerStyle}>
      <div style={containerStyle}>
        
        {/* Колона 1: Лого */}
        <div style={columnStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
            <div style={{ 
              background: brandGold, width: '32px', height: '32px', borderRadius: '6px', 
              display: 'flex', alignItems: 'center', justifyContent: 'center', 
              fontWeight: 'bold', color: brandBlue, fontSize: '20px'
            }}>е</div>
            <span style={{ fontWeight: 'bold', fontSize: '24px' }}>е-Влада</span>
          </div>
          <p style={{ fontSize: '14px', opacity: 0.7, lineHeight: '1.8', maxWidth: '300px' }}>
            Официјален портал на Владата на Република Северна Македонија.
          </p>
        </div>

        {/* Колона 2: МЕНИ СО ТОЧНИ ЛИНКОВИ */}
        <div style={{ ...columnStyle, alignItems: 'center' }}>
          <h4 style={{ color: brandGold, fontSize: '18px', fontWeight: 'bold', marginBottom: '25px', textTransform: 'uppercase' }}>Мени</h4>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            
            <Link to="/" style={linkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
              Почетна
            </Link>
            
            <Link to="/faq" style={linkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
              ЧПП
            </Link>
            
            <Link to="/services" style={linkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
              Услуги
            </Link>
            
            <Link to="/ai" style={linkStyle} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
              АИ Чат
            </Link>

          </div>
        </div>

        {/* Колона 3: Контакт */}
        <div style={{ ...columnStyle, alignItems: 'flex-start' }}>
          <h4 style={{ color: brandGold, fontSize: '18px', fontWeight: 'bold', marginBottom: '25px', textTransform: 'uppercase' }}>Контакт</h4>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{...linkStyle, cursor: 'default', opacity: 1}}>
              <span style={{ fontSize: '18px', width: '20px', textAlign: 'center' }}>📞</span> +389 2 3145 100
            </div>
            <div style={{...linkStyle, cursor: 'default', opacity: 1}}>
              <span style={{ fontSize: '18px', width: '20px', textAlign: 'center' }}>✉️</span> info@vlada.gov.mk
            </div>
            <div style={{...linkStyle, cursor: 'default', opacity: 1}}>
              <span style={{ fontSize: '18px', width: '20px', textAlign: 'center' }}>📍</span> Илинденска б.б., Скопје
            </div>
          </div>
        </div>

      </div>

      <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.15)', padding: '30px 20px', textAlign: 'center', fontSize: '13px', opacity: 0.6 }}>
        © 2026 ВЛАДА НА РЕПУБЛИКА СЕВЕРНА МАКЕДОНИЈА. СИТЕ ПРАВА СЕ ЗАДРЖАНИ.
      </div>
    </footer>
  );
}
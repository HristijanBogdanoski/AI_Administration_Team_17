import React from 'react';
import { useNavigate } from 'react-router-dom';

function Footer() {
    const navigate = useNavigate();

    return (
        <div className="ninth-container">
            <div className="footer-top">
                <div className="footer-left-info">
                    <h2 style={{color: "rgb(212, 160, 23)", margin: 0}}>Потребна ви е помош?</h2>
                    <p style={{fontSize: "0.9rem", color: "rgb(147, 197, 253)"}}>Нашиот тим е достапен секој работен ден</p>
                </div>
                <div className="footer-center-contact">
                    <div className="contact-item">
                        <span className="label">Телефон</span>
                        <p className="value">+389 2 3145 100</p>
                    </div>
                    <div className="contact-item">
                        <span className="label">Работно Време</span>
                        <p className="value">Пон–Пет, 08:00–16:30</p>
                    </div>
                </div>
                <div className="footer-right-button">
                    <button className="footer-ai-btn" onClick={() => navigate('/chat')}>Отвори АИ Чет</button>
                </div>
            </div>
            <hr style={{borderColor: "rgba(255,255,255,0.1)", margin: "40px 0"}}/>
            <div className="footer-bottom">
                <div className="footer-brand">
                    <h3 style={{color: "rgb(212, 160, 23)", marginBottom: "15px"}}>БрзиУслуги</h3>
                    <p>Официјален портал на Владата на Република Северна Македонија за јавни услуги и информации.</p>
                </div>
                <div className="footer-links">
                    <h4>Брзи врски</h4>
                    <ul>
                        <li onClick={() => navigate("/")}>Дома</li>
                        <li onClick={() => navigate("/faq")}>ЧПП</li>
                        <li onClick={() => navigate("/services")}>Услуги</li>
                        <li onClick={() => navigate("/locations")}>Локација</li>
                        <li onClick={() => navigate("/chat")}>АИ Чет</li>
                    </ul>
                </div>
                <div className="footer-contact-details">
                    <h4>Контакт</h4>
                    <p>📞 +389 2 3145 100</p>
                    <p>✉️ info@vlada.gov.mk</p>
                    <p>📍 Илинденска б.б., Скопје</p>
                </div>
            </div>
            <div className="footer-copyright">
                <p>© 2026 Република Северна Македонија</p>
            </div>
        </div>
    );
}

export default Footer;

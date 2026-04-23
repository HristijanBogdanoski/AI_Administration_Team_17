import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const NAV_LINKS = [
    { label: 'Дома', path: '/' },
    { label: 'ЧПП', path: '/faq' },
    { label: 'Услуги', path: '/' },
    { label: 'Локација', path: '/locations' }, // ПРОМЕНЕТО ОД '/' ВО '/locations'
    { label: 'АИ Чет', path: '/chat' },
];

function Navbar() {
    const navigate = useNavigate();
    const location = useLocation();
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [userEmail, setUserEmail] = useState('');

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            setIsLoggedIn(true);
            try {
                const payload = JSON.parse(atob(token.split('.')[1]));
                setUserEmail(payload.sub || '');
            } catch { /* invalid token */ }
        }
    }, [location]);

    const handleLogout = () => {
        localStorage.removeItem('token');
        setIsLoggedIn(false);
        setUserEmail('');
        navigate('/login');
    };

    return (
        <>
            {/* Banner */}
            <div style={{ background: '#CE2028', padding: '9px 16px', textAlign: 'center' }}>
                <p style={{ color: '#fff', margin: 0, fontSize: '0.78rem', fontWeight: 500 }}>
                    Портал на Владата на Република Северна Македонија
                </p>
            </div>

            {/* Navbar */}
            <nav style={{
                background: '#1B3A6B', padding: '0 40px', display: 'flex',
                alignItems: 'center', justifyContent: 'space-between',
                height: 64, position: 'sticky', top: 0, zIndex: 100
            }}>
                {/* Logo */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }} onClick={() => navigate('/')}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, background: 'rgba(212,160,23,.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>🛡️</div>
                    <div>
                        <div style={{ color: '#D4A017', fontWeight: 700, fontSize: '1rem' }}>е-Влада</div>
                        <div style={{ color: '#93c5fd', fontSize: '0.68rem' }}>Јавни Услуги</div>
                    </div>
                </div>

                {/* Links */}
                <div style={{ display: 'flex', gap: 4 }}>
                    {NAV_LINKS.map(({ label, path }) => {
                        
                        const active = location.pathname === path;
                        
                        return (
                            <button
                                key={label}
                                onClick={() => navigate(path)}
                                style={{
                                    background: active ? '#D4A017' : 'none',
                                    border: 'none',
                                    color: active ? '#0f2044' : '#ececf0',
                                    fontSize: '17px', cursor: 'pointer',
                                    padding: '8px 15px', borderRadius: 8,
                                    fontWeight: active ? 700 : 400,
                                }}
                            >
                                {label}
                            </button>
                        );
                    })}
                </div>

                {/* Auth */}
                {isLoggedIn ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span style={{ color: '#93c5fd', fontSize: '0.85rem' }}>{userEmail}</span>
                        <button onClick={handleLogout} style={{ background: 'none', border: '1.5px solid #93c5fd', color: '#93c5fd', padding: '7px 16px', borderRadius: 8, fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem' }}>
                            Одјави се
                        </button>
                    </div>
                ) : (
                    <button
                        onClick={() => navigate('/login')}
                        style={{
                            background: '#D4A017',
                            color: '#0f2044', border: 'none', padding: '9px 22px',
                            borderRadius: 8, fontWeight: 700, cursor: 'pointer', fontSize: '17px'
                        }}
                    >
                        Најава
                    </button>
                )}
            </nav>
        </>
    );
}

export default Navbar;
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const NAV_LINKS = [
    { label: 'Дома', path: '/' },
    { label: 'ЧПП', path: '/faq' },
    { label: 'Услуги', path: '/services' },
    { label: 'Локација', path: '/locations' },
    { label: 'АИ Чет', path: '/chat' },
];

function Navbar() {
    const navigate = useNavigate();
    const location = useLocation();
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [userEmail, setUserEmail] = useState('');
    const [isAdmin, setIsAdmin] = useState(false);
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            setIsLoggedIn(true);
            try {
                const payload = JSON.parse(atob(token.split('.')[1]));
                setUserEmail(payload.sub || '');
                setIsAdmin(payload.role === 'admin');
            } catch { /* invalid token */ }
        } else {
            setIsLoggedIn(false);
            setIsAdmin(false);
        }
    }, [location]);

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 10);
        window.addEventListener('scroll', onScroll);
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    const handleLogout = () => {
        localStorage.removeItem('token');
        setIsLoggedIn(false);
        setUserEmail('');
        navigate('/login');
    };

    const allLinks = [
        ...NAV_LINKS,
        ...(isAdmin ? [{ label: 'Корисници', path: '/admin/users' }] : []),
    ];

    return (
        <>
            {/* Top government banner */}
            <div style={{
                background: 'linear-gradient(90deg, #b91c1c 0%, #CE2028 50%, #b91c1c 100%)',
                padding: '7px 24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 10,
            }}>
                <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.72rem' }}>🏛️</span>
                <p style={{ color: '#fff', margin: 0, fontSize: '0.76rem', fontWeight: 500, letterSpacing: '0.03em' }}>
                    Портал на Владата на Република Северна Македонија
                </p>
            </div>

            {/* Main navbar */}
            <nav style={{
                background: scrolled
                    ? 'rgba(15,32,68,0.97)'
                    : 'linear-gradient(135deg, #0f2044 0%, #1B3A6B 60%, #1a4a8a 100%)',
                backdropFilter: 'blur(12px)',
                padding: '0 36px',
                display: 'grid',
                gridTemplateColumns: '200px 1fr auto',
                alignItems: 'center',
                height: 66,
                position: 'sticky',
                top: 0,
                zIndex: 100,
                boxShadow: scrolled ? '0 4px 24px rgba(0,0,0,0.35)' : '0 2px 12px rgba(0,0,0,0.2)',
                transition: 'background 0.3s, box-shadow 0.3s',
                borderBottom: '1px solid rgba(255,255,255,0.07)',
            }}>
                {/* Logo */}
                <div
                    onClick={() => navigate('/')}
                    style={{ display: 'flex', alignItems: 'center', gap: 11, cursor: 'pointer' }}
                >
                    <div style={{
                        width: 42, height: 42, borderRadius: 12,
                        background: 'linear-gradient(135deg, rgba(212,160,23,0.3) 0%, rgba(212,160,23,0.1) 100%)',
                        border: '1px solid rgba(212,160,23,0.35)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20,
                        boxShadow: '0 2px 8px rgba(212,160,23,0.2)',
                    }}>🛡️</div>
                    <div>
                        <div style={{ color: '#D4A017', fontWeight: 800, fontSize: '1.05rem', letterSpacing: '-0.01em' }}>е-Влада</div>
                        <div style={{ color: '#93c5fd', fontSize: '0.66rem', letterSpacing: '0.04em', opacity: 0.85 }}>ЈАВНИ УСЛУГИ</div>
                    </div>
                </div>

                {/* Nav links */}
                <div style={{ display: 'flex', gap: 2, justifyContent: 'center', alignItems: 'center' }}>
                    {allLinks.map(({ label, path }) => {
                        const active = location.pathname === path;
                        return (
                            <button
                                key={label}
                                onClick={() => navigate(path)}
                                className={`nav-link-btn${active ? ' active' : ''}`}
                            >
                                {label}
                            </button>
                        );
                    })}
                </div>

                {/* Auth */}
                {isLoggedIn ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'flex-end', flexWrap: 'nowrap', minWidth: 0 }}>
                        <button
                            onClick={() => navigate('/profile')}
                            style={{
                                background: 'rgba(255,255,255,0.06)',
                                border: '1px solid rgba(147,197,253,0.25)',
                                borderRadius: 8,
                                color: '#93c5fd',
                                fontSize: '0.78rem',
                                cursor: 'pointer',
                                padding: '6px 12px',
                                flexShrink: 0,
                                whiteSpace: 'nowrap',
                                transition: 'background 0.2s',
                            }}
                            onMouseOver={e => e.currentTarget.style.background = 'rgba(255,255,255,0.12)'}
                            onMouseOut={e => e.currentTarget.style.background = 'rgba(255,255,255,0.06)'}
                        >
                            👤 {userEmail}
                        </button>
                        <button
                            onClick={handleLogout}
                            style={{
                                background: 'none',
                                border: '1.5px solid rgba(147,197,253,0.4)',
                                color: '#93c5fd',
                                padding: '6px 14px',
                                borderRadius: 8,
                                fontWeight: 600,
                                cursor: 'pointer',
                                fontSize: '0.82rem',
                                flexShrink: 0,
                                whiteSpace: 'nowrap',
                                transition: 'all 0.2s',
                            }}
                            onMouseOver={e => { e.currentTarget.style.background = 'rgba(147,197,253,0.1)'; e.currentTarget.style.borderColor = '#93c5fd'; }}
                            onMouseOut={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.borderColor = 'rgba(147,197,253,0.4)'; }}
                        >
                            Одјави се
                        </button>
                    </div>
                ) : (
                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                        <button
                            onClick={() => navigate('/login')}
                            style={{
                                background: 'linear-gradient(135deg, #D4A017 0%, #f0b429 100%)',
                                color: '#0f2044',
                                border: 'none',
                                padding: '9px 22px',
                                borderRadius: 9,
                                fontWeight: 700,
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                boxShadow: '0 2px 10px rgba(212,160,23,0.35)',
                                transition: 'transform 0.15s, box-shadow 0.15s',
                            }}
                            onMouseOver={e => { e.currentTarget.style.transform = 'translateY(-1px)'; e.currentTarget.style.boxShadow = '0 4px 16px rgba(212,160,23,0.5)'; }}
                            onMouseOut={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 2px 10px rgba(212,160,23,0.35)'; }}
                        >
                            Најава
                        </button>
                    </div>
                )}
            </nav>
        </>
    );
}

export default Navbar;
import React from 'react';

export default function Services() {
  const uslugi = [
    { id: 1, title: "Лични документи", desc: "Пасоши, лични карти, возачки дозволи." },
    { id: 2, title: "Даноци и финансии", desc: "Даночни пријави, УЈП услуги, плаќања." },
    { id: 3, title: "Здравство", desc: "Мој Термин, здравствено осигурување." }
  ];

  return (
    <div className="container" style={{ padding: '40px 0' }}>
      <h1 style={{ color: 'var(--color-primary)' }}>Нашите Услуги</h1>
      
    </div>
  );
}
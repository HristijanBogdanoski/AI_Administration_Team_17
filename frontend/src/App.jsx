import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';
import Home from './pages/Home';
import FAQ from './pages/FAQ'; 
import AIChat from './pages/AI'; // Го користиме AIChat како име за компонентата од фајлот AI.jsx
import Services from './pages/Services'; 
import TopBanner from './components/layout/TopBanner';
import './styles/globals.css';

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        
        <TopBanner /> 
        <Navbar />
        
        <main style={{ flex: 1 }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/faq" element={<FAQ />} />
            
            {/* Оваа рута сега ќе ја отвора страната за АИ Чат */}
            <Route path="/ai" element={<AIChat />} />
            
            <Route path="/services" element={<Services />} />
            
            {/* Оваа линија секогаш треба да биде последна во Routes */}
            <Route path="*" element={<Home />} />
          </Routes>
        </main>
        
        <Footer />
      </div>
    </BrowserRouter>
  );
}
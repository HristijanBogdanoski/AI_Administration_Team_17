import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';
import Home from './pages/Home';
import FAQ from './pages/FAQ'; 
import AIChat from './pages/AI'; // <--- ДОДАЈ ГО ОВА (провери дали патеката е точна)
import Services from './pages/Services'; // <--- ДОДАЈ ГО ОВА
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
            
            {/* ГИ ДОДАВАМЕ ОВИЕ ДВЕ ЛИНИИ ЗА ДА РАБОТАТ ЛИНКОВИТЕ: */}
            <Route path="/ai" element={<AIChat />} />
            <Route path="/services" element={<Services />} />
            
            {/* Оваа линија ги „фаќа“ сите грешни линкови и ги враќа дома */}
            <Route path="*" element={<Home />} />
          </Routes>
        </main>
        
        <Footer />
      </div>
    </BrowserRouter>
  );
}
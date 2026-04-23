import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Login from './pages/Login';
import AiChat from './pages/AiChat';
import FAQ from './pages/FAQ';
import Locations from './pages/Locations'

function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path='/locations' element={<Locations/>}/>
          <Route path="/chat" element={<AiChat />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
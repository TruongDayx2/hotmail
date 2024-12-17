
import './App.css'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';

import HotMail from './pages/HotMail';
import TestDirection from './pages/TestDirection';
import ToolPhanHuong from './pages/ToolPhanHuong';
import TestHand from './pages/TestHand';

const App = () =>(
  <div>
  <Router>
    
    <Header />
    <main style={{ padding: '20px' }}>
      <Routes>
        <Route path="/" element={<HotMail />} />
        <Route path="/direction" element={<TestDirection />} />
        <Route path="/hand" element={<TestHand />} />
        <Route path="/phanHuong" element={<ToolPhanHuong />} />
      </Routes>
    </main>
  </Router>
  </div>
)

export default App

import { Link } from 'react-router-dom';

const Header = () => (
  <nav style={{ display: 'flex', justifyContent:'space-around', padding: '10px', background: '#e3e3e3' }}>
    <Link to="/">Hotmail</Link>
    <Link to="/direction">Dự đoán Hướng</Link>
    <Link to="/hand">Dự đoán Tay</Link>
    <Link to="/phanHuong">Phân Hướng</Link>
  </nav>
);

export default Header;

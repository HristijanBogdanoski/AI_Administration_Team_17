import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';

function Layout() {
    return (
        <div style={{ minHeight: '100vh', background: '#F4F6FA', display: 'flex', flexDirection: 'column' }}>
            <Navbar />
            <Outlet />
        </div>
    );
}

export default Layout;
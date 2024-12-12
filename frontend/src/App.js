import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Main from "./components/main.js";
import Front from "./components/front.js"

function App() {

  return (
    <Router>
        <div>
            <Routes>
              <Route path="/" element={<Main />} />
              <Route path="/front" element={<Front />} />
            </Routes>
        </div>
    </Router>
  );
}

export default App;

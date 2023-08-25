import "./App.css";
import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Ml from "./pages/Ml";
import Documentation from "./pages/Documentation";
import Ppt from "./pages/Ppt";

export default function App() {
  return (
    <div>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/ml" element={<Ml />} />
          <Route path="/doc" element={<Documentation />} />
          <Route path="/ppt" element={<Ppt />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

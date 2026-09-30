import { BrowserRouter, Routes, Route } from "react-router-dom";
import { RoleProvider } from "./context/RoleContext";
import { ThemeProvider } from "./context/ThemeContext";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import HomePage from "./pages/HomePage";
import PatientPortalPage from "./pages/PatientPortalPage";
import DoctorPortalPage from "./pages/DoctorPortalPage";
import CalibrationPage from "./pages/CalibrationPage";
import CaseRegistryPage from "./pages/CaseRegistryPage";

export default function App() {
  return (
    <ThemeProvider>
      <RoleProvider>
        <BrowserRouter>
          <div className="flex flex-col min-h-screen bg-[#f8fafc] text-slate-800 dark:bg-[#01151d] dark:text-slate-100 transition-colors duration-200">
            <Navbar />
            <main className="flex-1">
            <Routes>
              <Route path="/"            element={<HomePage />} />
              <Route path="/portal"      element={<PatientPortalPage />} />
              <Route path="/doctor"      element={<DoctorPortalPage />} />
              <Route path="/calibration" element={<CalibrationPage />} />
              <Route path="/cases"       element={<CaseRegistryPage />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </BrowserRouter>
    </RoleProvider>
  </ThemeProvider>
  );
}

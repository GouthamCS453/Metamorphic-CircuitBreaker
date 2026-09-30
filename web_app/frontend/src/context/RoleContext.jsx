// src/context/RoleContext.jsx
import { createContext, useContext, useState } from "react";

const RoleContext = createContext(null);

export const ROLES = {
  PATIENT: "patient",
  DOCTOR: "doctor",
  ADMIN: "admin",
};

export function RoleProvider({ children }) {
  const [role, setRole] = useState(ROLES.PATIENT);
  return (
    <RoleContext.Provider value={{ role, setRole }}>
      {children}
    </RoleContext.Provider>
  );
}

export const useRole = () => useContext(RoleContext);

import React from "react";
import Navbar from "../Components/Navbar";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from "recharts";

const HeartFailureAnalysis = () => {
  // Sample analysis data for heart failure prediction (replace with your actual data)
  const data = [
    { category: "Age", survived: 50, notSurvived: 100 },
    { category: "Gender", survived: 70, notSurvived: 80 },
    { category: "SmokingStatus", survived: 30, notSurvived: 120 },
    { category: "Diabetes", survived: 30, notSurvived: 120 },
    { category: "Anameia", survived: 30, notSurvived: 120 },
    { category: "HighBloodPressure", survived: 30, notSurvived: 120 },
    // Add more data...
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <Navbar />
      <div className="analysis">
        <h2 style={{ textAlign: "center" }}>Heart Failure Prediction Analysis</h2>
        <div style={{ textAlign: "center" }}>
          <BarChart width={600} height={400} data={data}>
            <XAxis dataKey="category" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="survived" fill="#8884d8" name="Survived" />
            <Bar dataKey="notSurvived" fill="#82ca9d" name="NotSurvived" />
          </BarChart>
        </div>
      </div>
    </div>
  );
};

export default HeartFailureAnalysis;

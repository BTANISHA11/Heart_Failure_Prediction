import React from "react";
import Navbar from "../Components/Navbar";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from "recharts";

const HeartFailureAnalysis = () => {
  // Sample analysis data for heart failure prediction (replace with your actual data)
  const data = [
    { category: "Age", positiveCases: 50, negativeCases: 100 },
    { category: "Gender", positiveCases: 70, negativeCases: 80 },
    { category: "Smoking Status", positiveCases: 30, negativeCases: 120 },
    // Add more data...
  ];

  return (
    <div>
      <Navbar />
      <div className="analysis">
        <h2>Heart Failure Prediction Analysis</h2>
        <BarChart width={600} height={400} data={data}>
          <XAxis dataKey="category" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="positiveCases" fill="#8884d8" name="Positive Cases" />
          <Bar dataKey="negativeCases" fill="#82ca9d" name="Negative Cases" />
        </BarChart>
      </div>
    </div>
  );
};

export default HeartFailureAnalysis;

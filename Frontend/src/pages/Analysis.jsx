import React from "react";
import Navbar from "../Components/Navbar";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from "recharts";

import heartImage from "../Assets/age_graph1.png"; // Replace with the actual path to your image
import heartImage1 from "../Assets/smoker.png";
import heartImage2 from "../Assets/diabetes.png";
import heartImage3 from "../Assets/anaemia.png";


const HeartFailureAnalysis = () => {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <Navbar />
      <div className="analysis">
        <h2 style={{ textAlign: "center" }}>Heart Failure Prediction Analysis</h2>
        <div style={{ textAlign: "center" }}>
          <img src={heartImage} alt="Heart" style={{ maxWidth: "100%", height: "auto" }} />
          <img src={heartImage1} alt="Heart" style={{ maxWidth: "100%", height: "auto" }} />
          <img src={heartImage2} alt="Heart" style={{ maxWidth: "100%", height: "auto" }} />
          <img src={heartImage3} alt="Heart" style={{ maxWidth: "100%", height: "auto" }} />

        </div>
      </div>
    </div>
  );
};

export default HeartFailureAnalysis;

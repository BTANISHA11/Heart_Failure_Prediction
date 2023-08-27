import React from "react";
import Navbar from "../Components/Navbar";
export default function Documentation() {
  // const driveLink = "https://drive.google.com/file/d/1TFMw8NpEJ4ICCO4MkeTmUcA2HYIp1qMt/view?usp=sharing";
// export default function Documentation() {
  return (
    <div>
      <Navbar />
      <div className="doc">
        <iframe
          className="doc"
          src="https://research-paper.tiiny.site/"
          // src="ML\HEART FAILURE PREDICTION.ipynb"
          // src="HEART FAILURE PREDICTION.html"
          // src={driveLink}
          width="1700"
          height="850"
          // margin=""
          title="DOC"
        ></iframe>
      </div>
    </div>
  );
}

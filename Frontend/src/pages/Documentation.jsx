import React from "react";
import Navbar from "../Components/Navbar";

export default function Documentation() {
  return (
    <div>
      <Navbar />
      <div className="doc">
        <iframe
          className="doc"
          src="https://research-paper.tiiny.site/"
          width="1700"
          height="850"
          // margin=""
          title="DOC"
        ></iframe>
      </div>
    </div>
  );
}

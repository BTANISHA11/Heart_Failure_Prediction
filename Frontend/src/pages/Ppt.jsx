import React from "react";
import Navbar from "../Components/Navbar";

export default function Ppt() {
  return (
    <div>
      <Navbar />
      <div className="ppt">
        <iframe
          src="https://www.slideshare.net/secret/32YUrCLs2bPeSd"
          width="1450"
          height="800"
          frameborder="0"
          marginwidth="0"
          marginheight="0"
          scrolling="no"
          title="PPT"
        ></iframe>
      </div>
    </div>
  );
}

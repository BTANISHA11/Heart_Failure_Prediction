import React from "react";
import Navbar from "../Components/Navbar";

export default function Ppt() {
  return (
    <div>
      <Navbar />
      <div className="ppt">
        <iframe
          src="https://www.slideshare.net/slideshow/embed_code/key/HusIn6BFM8yRjm?hostedIn=slideshare&page=upload"
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

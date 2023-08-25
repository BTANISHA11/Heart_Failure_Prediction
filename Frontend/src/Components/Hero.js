import React from "react";
import BannerBackground from "../Assets/blob.svg";
import BannerImage from "../Assets/hero image.png";
import Navbar from "./Navbar";
// import { FiArrowRight } from "react-icons/fi";
import {Link} from "react-router-dom"
const Home = () => {
  return (
    <div className="home-container">
      <Navbar />
      <div className="home-banner-container">
        <div className="home-bannerImage-container">
          <img src={BannerBackground} alt="" />
        </div>
        <div className="home-text-section">
          <h1 className="primary-heading">
            Empowering Health through Early Awareness
          </h1>
          <p className="primary-text">
            Welcome to Heartify, a Risk Assessment tool, designed to help
            individuals assess their risk factors for heart failure.
          </p>
          <button className="secondary-button">
            <Link to="/"> Give it a try!</Link>
            {/* {<Link to="/ml">Give ti a try</Link>} */}
            {/* Give it a try <FiArrowRight />{" "} */}
          </button>
        </div>
        <div className="home-image-section">
          <img src={BannerImage} alt="" />
        </div>
      </div>
    </div>
  );
};

export default Home;

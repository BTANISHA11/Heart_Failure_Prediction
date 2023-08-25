import React from "react";
import AboutBackground from "../Assets/hero second image.png";

const About = () => {
  return (
    <div className="about-section-container">
      <div className="about-background-image-container">
        <img src={AboutBackground} alt="" />
      </div>
      <div className="about-section-image-container">
        {/* <img src={AboutBackgroundImage} alt="" /> */}
      </div>
      <div className="about-section-text-container">
        <p className="primary-subheading">About</p>
        <h1 className="primary-heading">
          Stay Ahead of Heart Failure: Predicting Your Health, Saving Your Heart
        </h1>
        <p className="primary-text">
          By evaluating various aspects of your lifestyle, medical history, and
          physiological indicators, our predictive model can estimate your
          likelihood of developing heart failure.
        </p>
        <p className="primary-text">
          Remember, this tool provides an informative assessment, but it's
          important to consult with a healthcare professional for an accurate
          diagnosis and personalized advice.
        </p>
        <div className="about-buttons-container">
          <button className="secondary-button">Learn More</button>
        </div>
      </div>
    </div>
  );
};

export default About;

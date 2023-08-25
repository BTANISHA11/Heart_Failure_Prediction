import { useState, useRef, useEffect } from "react";
import axios from "axios";
import Navbar from "../Components/Navbar";

export default function Ml() {
  const bottomRef = useRef(null);
  const [age, setAge] = useState();
  const [anaemia, setAnaemia] = useState("");
  const [creatinine_phosphokinase, setCreatinine_phosphokinase] = useState("");
  const [diabetes, setdiabetes] = useState("");
  const [ejection_fraction, setejection_fraction] = useState("");
  const [high_blood_pressure, sethigh_blood_pressure] = useState("");
  const [platelets, setplatelets] = useState("");
  const [serum_creatinine, setserum_creatinine] = useState("");
  const [serum_sodium, setserum_sodium] = useState("");
  const [sex, setSex] = useState("");
  const [smoking, setSmoking] = useState("");
  const [time, setTime] = useState("");
  const [prediction, setPrediction] = useState();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    // alert("post");
    // setLoaded(false)
    setAge(Number(age));
    setAnaemia(Number(anaemia));
    setCreatinine_phosphokinase(Number(creatinine_phosphokinase));
    setdiabetes(Number(diabetes));
    setejection_fraction(Number(ejection_fraction));
    sethigh_blood_pressure(Number(high_blood_pressure));
    setplatelets(Number(platelets));
    setserum_creatinine(Number(serum_creatinine));
    setserum_sodium(Number(serum_sodium));
    setSex(Number(sex));
    setSmoking(Number(smoking));
    setTime(Number(time));

    await axios
      .post("https://heart-failure-api-v2.onrender.com/prediction", {
        age,
        anaemia,
        creatinine_phosphokinase,
        diabetes,
        ejection_fraction,
        high_blood_pressure,
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        smoking,
        time,
      })
      .then((response) => {
        // setLoaded(true)
        setPrediction(response.data["Prediction"]);
        console.log(response.data["Prediction"]);
      })
      .catch((error) => {
        console.log(error);
      });
  };

  return (
    <div className="ml-form">
      <Navbar />
      <form>
        <h1>Fill Details</h1>
        <label className="form-fields">
          Age:
          <input
            className="input-container"
            type="int"
            value={age}
            onChange={(e) => setAge(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Anaemia:
          <input
            className="input-container"
            type="int"
            value={anaemia}
            onChange={(e) => setAnaemia(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Creatinine Phosphokinase:
          <input
            className="input-container"
            type="int"
            value={creatinine_phosphokinase}
            onChange={(e) => setCreatinine_phosphokinase(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Diabetes:
          <input
            className="input-container"
            type="int"
            value={diabetes}
            onChange={(e) => setdiabetes(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Ejection Fraction:
          <input
            className="input-container"
            type="int"
            value={ejection_fraction}
            onChange={(e) => setejection_fraction(e.target.value)}
          />
        </label>
        <label className="form-fields">
          High Blood Pressure:
          <input
            className="input-container"
            type="int"
            value={high_blood_pressure}
            onChange={(e) => sethigh_blood_pressure(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Platelets:
          <input
            className="input-container"
            type="int"
            value={platelets}
            onChange={(e) => setplatelets(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Serum Creatinine:
          <input
            className="input-container"
            type="int"
            value={serum_creatinine}
            onChange={(e) => setserum_creatinine(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Serum Sodium:
          <input
            className="input-container"
            type="int"
            value={serum_sodium}
            onChange={(e) => setserum_sodium(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Sex:
          <input
            className="input-container"
            type="int"
            value={sex}
            onChange={(e) => setSex(e.target.value)}
          />
        </label>
        <label className="form-fields">
          Smoking:
          <input
            className="input-container"
            type="int"
            value={smoking}
            onChange={(e) => setSmoking(e.target.value)}
          />
        </label>

        <label className="form-fields">
          Follow up time period:
          <input
            className="input-container"
            type="int"
            value={time}
            onChange={(e) => setTime(e.target.value)}
          />
        </label>

        <input
          className="secondary-button"
          type="submit"
          onClick={handleSubmit}
        />
        <div ref={bottomRef}/>
        <h2>
          Chances of Heart Failure: {Math.floor(prediction * 1000) / 1000}
        </h2>
      </form>
    </div>
  );
}

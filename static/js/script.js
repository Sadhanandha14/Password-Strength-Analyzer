

let currentController = null;
let requestSeq = 0;


function handlePasswordInput() {
    const input = document.getElementById("password");
    const password = input.value.trim();
    const hintText = document.getElementById("hint-text");
    const strengthBar = document.querySelector(".strength-bar");
    const strengthLabel = document.getElementById("strength-label");
    const cards = document.querySelector(".cards");
    const suggestions = document.querySelector(".suggestions");

    requestSeq++; 


    if (!password) {

        hintText.style.opacity = "1";
        hintText.style.visibility = "visible";
        strengthBar.style.display = "none";
        strengthLabel.style.display = "none";
        cards.style.display = "none";
        suggestions.style.display = "none";

        resetUI();
        return;
    }

    strengthBar.style.display = "block";
    strengthLabel.style.display = "block";
    cards.style.display = "grid";
    suggestions.style.display = "block";
    hintText.style.opacity = "0";
    hintText.style.visibility = "hidden";
    analyzePasswordLive(password, requestSeq);
}


async function analyzePasswordLive(password, seq) {

    
    if (currentController) {
        currentController.abort();
    }

    currentController = new AbortController();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ password }),
            signal: currentController.signal
        });

        
        if (seq !== requestSeq) return;

        if (!response.ok) {
            resetUI();
            return;
        }

        const data = await response.json();

        
        const currentInput = document.getElementById("password").value.trim();
        if (!currentInput) return;


const score = data.score;

let strengthText =
    score >= 70 ? "STRONG" :
    score >= 50 ? "GOOD" :
    "WEAK";


document.getElementById("strength-fill").style.width = score + "%";
document.getElementById("strength-label").innerText = strengthText;


document.getElementById("score").innerText = score;
document.getElementById("score-text").innerText = strengthText;

const circle = document.querySelector(".progress-circle .progress");
const radius = 52;
const circumference = 2 * Math.PI * radius;
const offset = circumference - (score / 100) * circumference;

circle.style.strokeDasharray = circumference;
circle.style.strokeDashoffset = offset;
circle.style.stroke = "#6366f1";

        document.getElementById("time").innerText = data.time_to_crack;
        document.getElementById("entropy").innerText =
            `${Math.round(data.entropy_bits)} bits`;
        document.getElementById("confidence").innerText =
            `${Math.round(data.confidence)}%`;


        const suggestionBox = document.getElementById("suggestion");
        if (Array.isArray(data.explanation) && data.explanation.length) {
            suggestionBox.innerHTML = data.explanation
                .map(item => `<div>✔ ${item}</div>`)
                .join("");
        } else {
            suggestionBox.innerHTML = `<div>✔ Great password structure!</div>`;
        }

    } catch (err) {
        if (err.name !== "AbortError") {
            console.error("Prediction error:", err);
        }
    }
}


function resetUI() {
    document.getElementById("strength-fill").style.width = "0%";
    document.getElementById("strength-label").innerText = "—";

    document.getElementById("score").innerText = "0";
    document.getElementById("score-text").innerText = "—";

    document.getElementById("time").innerText = "—";
    document.getElementById("entropy").innerText = "—";
    document.getElementById("confidence").innerText = "—";
    document.getElementById("suggestion").innerHTML = "—";

    const circle = document.querySelector(".progress-circle .progress");
    if (circle) {
        const radius = 52;
        const circumference = 2 * Math.PI * radius;
        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = circumference;
        circle.style.stroke = "#e5e7eb";
    }
}


document.addEventListener("DOMContentLoaded", () => {
    resetUI();

    document.getElementById("hint-text").style.visibility = "visible";
    // Hide UI on load
    document.querySelector(".strength-bar").style.display = "none";
    document.getElementById("strength-label").style.display = "none";
    document.querySelector(".cards").style.display = "none";
    document.querySelector(".suggestions").style.display = "none";

    initSketchEyeToggle();
});


function initSketchEyeToggle() {
    const passwordInput = document.getElementById("password");
    const toggleBtn = document.getElementById("toggle-password");
    const eyeOpen = document.getElementById("eye-open");
    const eyeClosed = document.getElementById("eye-closed");

    if (!toggleBtn) return;

    toggleBtn.addEventListener("click", () => {
        const isHidden = passwordInput.type === "password";
        passwordInput.type = isHidden ? "text" : "password";
        eyeOpen.classList.toggle("hidden", !isHidden);
        eyeClosed.classList.toggle("hidden", isHidden);
    });
}

// // // ================= CHATBOT JAVASCRIPT =================

// // const chatbotToggler = document.querySelector(".chatbot-toggler");
// // const closeBtn = document.querySelector(".close-btn");
// // const chatbox = document.querySelector(".chatbox");
// // const chatInput = document.querySelector(".chat-input textarea");
// // const sendChatBtn = document.querySelector(".chat-input span");

// // let userMessage = null; // Variable to store user's message

// // const createChatLi = (message, className) => {
// //     // Create a chat <li> element with passed message and className
// //     const chatLi = document.createElement("li");
// //     chatLi.classList.add("chat", className);
// //     let chatContent = className === "outgoing" 
// //         ? `<p></p>` 
// //         : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
// //     chatLi.innerHTML = chatContent;
// //     chatLi.querySelector("p").textContent = message;
// //     return chatLi; // return chat <li> element
// // }

// // const generateResponse = (message) => {
// //     const lower = message.toLowerCase();
// //     let responseText = "";

// //     // === YOUR LOGIC HERE ===
// //     if (lower.includes("hi") || lower.includes("hello")) {
// //         responseText = "Hello! How can I help you today?";
// //     } else {
// //         // Run Password Logic
// //         responseText = checkPasswordStrength(message);
// //     }
// //     // =======================

// //     return responseText;
// // }

// // const checkPasswordStrength = (password) => {
// //     let strength = 0;
// //     let tips = [];

// //     if (password.length >= 8) strength++; else tips.push("Make it longer (8+ chars)");
// //     if (/[A-Z]/.test(password)) strength++; else tips.push("Add Uppercase (A-Z)");
// //     if (/[a-z]/.test(password)) strength++; else tips.push("Add Lowercase (a-z)");
// //     if (/[0-9]/.test(password)) strength++; else tips.push("Add Numbers (0-9)");
// //     if (/[^A-Za-z0-9]/.test(password)) strength++; else tips.push("Add Symbols (!@#)");

// //     if (strength === 5) return "Strong password! ✅";
    
// //     // Join tips with newlines for the CSS 'pre-wrap' to handle
// //     return `Weak Password ⚠️\n\nTips:\n• ${tips.join("\n• ")}`;
// // }

// // const handleChat = () => {
// //     userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
// //     if (!userMessage) return;

// //     // Clear the input textarea and set its height to default
// //     chatInput.value = "";

// //     // Append the user's message to the chatbox
// //     chatbox.appendChild(createChatLi(userMessage, "outgoing"));
// //     chatbox.scrollTo(0, chatbox.scrollHeight);

// //     // Display "Thinking..." or just delay
// //     setTimeout(() => {
// //         const botResponse = generateResponse(userMessage);
// //         const incomingChatLi = createChatLi(botResponse, "incoming");
// //         chatbox.appendChild(incomingChatLi);
// //         chatbox.scrollTo(0, chatbox.scrollHeight);
// //     }, 600);
// // }

// // chatInput.addEventListener("keydown", (e) => {
// //     // If Enter key is pressed without Shift key and the window width is > 800px, handle the chat
// //     if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
// //         e.preventDefault();
// //         handleChat();
// //     }
// // });

// // sendChatBtn.addEventListener("click", handleChat);
// // closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
// // chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));





































































// // ================= CHATBOT JAVASCRIPT =================

// const chatbotToggler = document.querySelector(".chatbot-toggler");
// const closeBtn = document.querySelector(".close-btn");
// const chatbox = document.querySelector(".chatbox");
// const chatInput = document.querySelector(".chat-input textarea");
// const sendChatBtn = document.querySelector(".chat-input span");
// const clearBtn = document.querySelector(".clear-btn"); // New clear chat button

// let userMessage = null; // Variable to store user's message

// // Load chat history from localStorage
// window.addEventListener("load", () => {
//     const savedChats = JSON.parse(localStorage.getItem("chatHistory")) || [];
//     savedChats.forEach(chat => {
//         chatbox.appendChild(createChatLi(chat.message, chat.type));
//     });
//     chatbox.scrollTo(0, chatbox.scrollHeight);
// });

// // Save chat history
// const saveChat = (message, type) => {
//     let chats = JSON.parse(localStorage.getItem("chatHistory")) || [];
//     chats.push({ message, type });
//     localStorage.setItem("chatHistory", JSON.stringify(chats));
// };

// const createChatLi = (message, className) => {
//     // Create a chat <li> element with passed message and className
//     const chatLi = document.createElement("li");
//     chatLi.classList.add("chat", className);
//     let chatContent = className === "outgoing"
//         ? `<p></p>`
//         : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
//     chatLi.innerHTML = chatContent;
//     chatLi.querySelector("p").textContent = message;
//     return chatLi; // return chat <li> element
// }

// const generateResponse = (message) => {
//     const lower = message.toLowerCase();
//     let responseText = "";

//     // === BOT LOGIC ===
//     if (lower.includes("hi") || lower.includes("hello")) {
//         responseText = "Hello! 👋 How can I help you today?";
//     } else if (lower.includes("time")) {
//         responseText = `⏰ Current Time: ${new Date().toLocaleTimeString()}`;
//     } else if (lower.includes("date")) {
//         responseText = `📅 Today's Date: ${new Date().toLocaleDateString()}`;
//     } else if (lower.includes("joke")) {
//         const jokes = [
//             "😂 Why don’t skeletons fight each other? They don’t have the guts.",
//             "🤣 I told my computer I needed a break, and it said 'No problem — I’ll go to sleep.'",
//             "😆 Why was the math book sad? Because it had too many problems."
//         ];
//         responseText = jokes[Math.floor(Math.random() * jokes.length)];
//     } else {
//         // Run Password Logic
//         responseText = checkPasswordStrength(message);
//     }
//     // =======================

//     return responseText;
// }

// const checkPasswordStrength = (password) => {
//     let strength = 0;
//     let tips = [];

//     if (password.length >= 8) strength++; else tips.push("Make it longer (8+ chars)");
//     if (/[A-Z]/.test(password)) strength++; else tips.push("Add Uppercase (A-Z)");
//     if (/[a-z]/.test(password)) strength++; else tips.push("Add Lowercase (a-z)");
//     if (/[0-9]/.test(password)) strength++; else tips.push("Add Numbers (0-9)");
//     if (/[^A-Za-z0-9]/.test(password)) strength++; else tips.push("Add Symbols (!@#)");

//     if (strength === 5) return "Strong password! ✅🔒";
    
//     return `Weak Password ⚠️\n\nTips:\n• ${tips.join("\n• ")}`;
// }

// const handleChat = () => {
//     userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
//     if (!userMessage) return;

//     chatInput.value = ""; // Clear input

//     // Append the user's message to the chatbox
//     const outgoingChatLi = createChatLi(userMessage, "outgoing");
//     chatbox.appendChild(outgoingChatLi);
//     saveChat(userMessage, "outgoing");
//     chatbox.scrollTo(0, chatbox.scrollHeight);

//     // Show typing indicator
//     const typingLi = createChatLi("Typing...", "incoming");
//     chatbox.appendChild(typingLi);
//     chatbox.scrollTo(0, chatbox.scrollHeight);

//     // Delay bot response
//     setTimeout(() => {
//         chatbox.removeChild(typingLi); // Remove typing indicator
//         const botResponse = generateResponse(userMessage);
//         const incomingChatLi = createChatLi(botResponse, "incoming");
//         chatbox.appendChild(incomingChatLi);
//         saveChat(botResponse, "incoming");
//         chatbox.scrollTo(0, chatbox.scrollHeight);
//     }, 1000);
// }

// // Event listeners
// chatInput.addEventListener("keydown", (e) => {
//     if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
//         e.preventDefault();
//         handleChat();
//     }
// });

// sendChatBtn.addEventListener("click", handleChat);
// closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
// chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));

// // Clear chat history
// clearBtn.addEventListener("click", () => {
//     localStorage.removeItem("chatHistory");
//     chatbox.innerHTML = "";
// });














































// ================= CHATBOT JAVASCRIPT =================

const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const clearBtn = document.querySelector(".clear-btn"); // Clear chat button

let userMessage = null; // Variable to store user's message

// // Load chat history from localStorage
// window.addEventListener("load", () => {
//     const savedChats = JSON.parse(localStorage.getItem("chatHistory")) || [];
//     savedChats.forEach(chat => {
//         chatbox.appendChild(createChatLi(chat.message, chat.type));
//     });
//     chatbox.scrollTo(0, chatbox.scrollHeight);
// });

// Save chat history
const saveChat = (message, type) => {
    let chats = JSON.parse(localStorage.getItem("chatHistory")) || [];
    chats.push({ message, type });
    localStorage.setItem("chatHistory", JSON.stringify(chats));
};

const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent = className === "outgoing"
        ? `<p></p>`
        : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi;
}

// === INTENT PREDICTION LOGIC ===
const generateResponse = (message) => {
    const lower = message.toLowerCase().trim();
    let responseText = "";

    // Greeting detection
    if (lower.includes("hi") || lower.includes("hello") || lower.includes("hey")) {
        responseText = " Hello! How can I assist you today?";
    } 
    // Query detection (sentence with more than 3 words)
    else if (message.split(" ").length > 3) {
        responseText = ` That looks like a query.\nYou asked: "${message}"\nI’ll try to help with that!`;
    } 
    // Password detection (short string with mixed characters)
    else {
        responseText = checkPasswordStrength(message);
    }

    return responseText;
}

// === PASSWORD STRENGTH CHECKER ===
const checkPasswordStrength = (password) => {
    let strength = 0;
    let tips = [];

    if (password.length >= 8) strength++; else tips.push("Make it longer (8+ chars)");
    if (/[A-Z]/.test(password)) strength++; else tips.push("Add Uppercase (A-Z)");
    if (/[a-z]/.test(password)) strength++; else tips.push("Add Lowercase (a-z)");
    if (/[0-9]/.test(password)) strength++; else tips.push("Add Numbers (0-9)");
    if (/[^A-Za-z0-9]/.test(password)) strength++; else tips.push("Add Symbols (!@#)");

    if (strength === 5) {
        return " Strong password! Well done — it’s secure.";
    }

    // Efficient suggestions
    return `Weak Password\nSuggestions to improve:\n• ${tips.join("\n• ")}`;
}

// === HANDLE CHAT FLOW ===
const handleChat = () => {
    userMessage = chatInput.value.trim();
    if (!userMessage) return;

    chatInput.value = "";

    const outgoingChatLi = createChatLi(userMessage, "outgoing");
    chatbox.appendChild(outgoingChatLi);
    saveChat(userMessage, "outgoing");
    chatbox.scrollTo(0, chatbox.scrollHeight);

    const typingLi = createChatLi("Typing...", "incoming");
    chatbox.appendChild(typingLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);

    setTimeout(() => {
        chatbox.removeChild(typingLi);
        const botResponse = generateResponse(userMessage);
        const incomingChatLi = createChatLi(botResponse, "incoming");
        chatbox.appendChild(incomingChatLi);
        saveChat(botResponse, "incoming");
        chatbox.scrollTo(0, chatbox.scrollHeight);
    }, 1000);
}

// === EVENT LISTENERS ===
chatInput.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleChat();
    }
});

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));

// Clear chat history
clearBtn.addEventListener("click", () => {
    localStorage.removeItem("chatHistory");
    chatbox.innerHTML = "";
});

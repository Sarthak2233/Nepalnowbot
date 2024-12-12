import React, { useState } from 'react';
import { Link } from "react-router-dom";

function Main(){

    const [isVisible, setIsVisible] = useState(false);

  // Toggle the visibility of the root content
      const toggleVisibility = () => {
        setIsVisible(!isVisible);
  };

  const [messages, setMessages] = useState([]); // Store chat messages
    const [relatedQuestions, setRelatedQuestions] = useState([]); // Store related questions
    const [input, setInput] = useState(""); // Store input field value

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage = input.trim();
        setMessages((prev) => [...prev, { type: "user", text: userMessage }]);
        setInput(""); // Clear input field

        try {
            // Call the /query API
            const response = await fetch("http://localhost:8002/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_query: userMessage }),
            });

            const data = await response.json();
            console.log(data, "data")
            console.log(data.response, "Response data")


            if (data.response) {
                setMessages((prev) => [...prev, { type: "bot", text: data.response }]);
            }

            if (data.related_questions) {
                setRelatedQuestions(data.related_questions);
                console.log(data.related_questions, "related_questions")
            }
        } catch (error) {
            setMessages((prev) => [...prev, { type: "bot", text: "Error fetching response." }]);
        }
    };

    const handleRelatedQuestionClick = (question) => {
        setInput(question);
        sendMessage(); // Automatically send the question
    };


    return(
    <div className='landingPage'>
      {!isVisible && (
         <>
          <img src="/images/nepalnow.png" alt="Centered Image" className="centerImage" />
          <img src="/images/ntb.png" alt="NTB Image" className="ntbImage" />
        </>
       )}
      {/* Call-to-action Button */}
      <button id="cta-button" onClick={toggleVisibility}>
        <i className={`fas ${isVisible ? 'fa-times' : 'fa-comment'}`}></i>
      </button>

      {/* React Content */}

        <section id="root-content" className={`${isVisible ? 'show' : ''} main`}>
            <div className="view">
                <div className="bg">
                    <div className="line-container">
                        <div className="line"></div>
                        <div className="ask">ASK</div>
                        <div className="line"></div>
                    </div>
                    <div className="logo-section">
                        <div className="logo">
                            <img src="/images/nepalnow.png" alt="FAQ Icon" />
                        </div>
                        <div className="section">
                            News..  Stories... Updates....
                        </div>
                    </div>
                </div>
                <div className="icons">
                    <div className="icon-faq">
                        <img src="/images/faq.png" alt="FAQ Icon" />
                    </div>
                    <div className="icon-faq">
                        <img src="/images/plane.png" alt="Plane Icon" />
                    </div>
                    <div className="icon-faq">
                        <img src="/images/news.png" alt="News Icon" />
                    </div>
                    <div className="icon-faq">
                        <img src="/images/weather.png" alt="Plane Icon" />
                    </div>
                    <div className="icon-faq">
                        <img src="/images/tick.png" alt="Fifth Icon" />
                    </div>
                </div>
                <div className='says'>
                    <div className='dot'>

                    </div>
                    <p>nepalNOW says :-):-):-):</p>
                </div>
                <div className="info">
                     <div className="messages">
                            {messages.map((msg, index) => (
                                <div
                                    key={index}
                                    className={`message ${msg.type === "user" ? "user-message" : "bot-message"}`}
                                >
                                    {msg.type === "bot" ? (
                                        <div dangerouslySetInnerHTML={{ __html: msg.text }} />
                                    ) : (
                                        msg.text
                                    )}
                                </div>
                            ))}
                            {/* Related Questions Section */}
                            <div className="related-questions">
                                {relatedQuestions
                                    .filter((q) => q && q.question) // Filter out falsy values and empty questions
                                    .map((q, index) => (
                                        <button
                                            key={index}
                                            className="related-question"
                                            onClick={() => handleRelatedQuestionClick(q.question)}
                                        >
                                            {q.question}
                                        </button>
                                    ))}
                            </div>
                        </div>


                </div>

                <div className="input-container">
                    <i className="fas fa-microphone"></i>
                    <input type="text" placeholder="Ask your questions" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendMessage()} />
                    <i className="fas fa-paper-plane send-icon" onClick={sendMessage}></i>
                </div>
            </div>

        </section>

    </div>


    )
}
export default Main;
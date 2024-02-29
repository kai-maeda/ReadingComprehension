   // practice.js
const apiKey = '97b625403dmsh2c97be9f01dbfa3p103a76jsn9af481a186ef';
let testButton = document.getElementById("testbutton");
let email = document.getElementById('email');

// const savedWords = [
//     { word: "apple" },
//     { word: "banana" },
//     { word: "orange" }
// ];

function writeUserData(userId, email) {
    const db = getDatabase();
    const reference = ref(db, 'users/' + userId);
    set(reference, {
      email: email,
    });
    console.log("added to database")
  }


function updateWordList() {
    const savedWordsList = document.getElementById("saved-words");
    savedWordsList.innerHTML = "";

    savedWords.forEach(item => {
        const listItem = document.createElement("div");
        listItem.classList.add("word-item");

        // Display word
        const wordSpan = document.createElement("span");
        wordSpan.textContent = `${item.word}: `;
        listItem.appendChild(wordSpan);

        // Fetch definition from Words API
        fetch(`https://wordsapiv1.p.rapidapi.com/words/${item.word}`, {
            method: 'GET',
            headers: {
                'X-RapidAPI-Key': apiKey,
                'X-RapidAPI-Host': 'wordsapiv1.p.rapidapi.com',
            },
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Display definition
            const definitionSpan = document.createElement("span");
            definitionSpan.classList.add("word-definition");

            if (data && data.results && data.results.length > 0) {
                definitionSpan.textContent = data.results[0].definition;
            } else {
                definitionSpan.textContent = "Definition not found or unavailable.";
            }

            listItem.appendChild(definitionSpan);
            savedWordsList.appendChild(listItem);
        })
        .catch(error => {
            console.error('Error fetching definition:', error);
        });
    });
}

function getRandomWord() {
    return savedWords[Math.floor(Math.random() * savedWords.length)];
}

function practiceRandomWord() {
    const wordToPractice = getRandomWord();
    document.getElementById("word-to-practice").innerText = `Practice Word: ${wordToPractice.word}`;
    document.getElementById("result-message").innerText = ""; // Clear result message
}

function deleteWord(button) {
    var listItem = button.parentNode;
    listItem.parentNode.removeChild(listItem);
}
 
// Initial setup
updateWordList();
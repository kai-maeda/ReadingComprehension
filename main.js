document.addEventListener('DOMContentLoaded', function () {
  setupEventListeners();

  function setupEventListeners() {
      let email = document.getElementById('email');
      let password = document.getElementById('password');
      let loginButton = document.getElementById('login');
      let signupButton = document.getElementById('signup');
      let logoutButton = document.getElementById('logout');
      let logoutButton2 = document.getElementById('logout2');
      let homeButton = document.getElementById("home");
      let difficulty = document.getElementById("difficulty");

      if (loginButton) {
          loginButton.addEventListener('click', function () {
              login();
          });
      }

      if (signupButton) {
          signupButton.addEventListener('click', function () {
              signup();
          });
      }

      if (logoutButton) {
          logoutButton.addEventListener('click', function () {
              logout();
          });
      }
      if (logoutButton2) {
          logoutButton2.addEventListener('click', function () {
              logout();
          });
      }
      if (homeButton) {
          homeButton.addEventListener('click', function () {
              console.log('Home button clicked!');
              window.open('practice/practice.html', '_blank');
          });
      }
      if (difficulty) {
        difficulty.addEventListener('click', function() {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                const apiUrl = 'http://127.0.0.1:5000/word_count';
                const currentUrl = tabs[0].url;
                const data = {url: currentUrl};
                fetch(apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                })
                .then(response => response.json())
                .then(result => {
                    console.log(result);
                    if(result.difficulty == -1) {
                        document.getElementById('currentUrl').textContent = 'Sorry, score is unavailable for this webpage.';
                    } else {
                        document.getElementById('currentUrl').textContent = 'Reading Difficulty: ' + result.difficulty.toFixed(2);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
        });
      }
    }
    function updateCurrentUrl() {
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            const currentUrl = tabs[0].url;
            document.getElementById('currentUrl').textContent = 'Current URL: ' + currentUrl;
        });
    }
    updateCurrentUrl();

    // chrome.tabs.onActivated.addListener(function(activeInfo) {
    //     // Call the function when the active tab changes
    //     updateCurrentUrl();
    // });
    chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
        if (changeInfo.status === 'loading') {
            updateCurrentUrl();
        }
    });

  function login() {
      let e = email.value;
      let p = password.value;
      firebase.auth().signInWithEmailAndPassword(e, p)
          .catch((error) => {
              console.error("Login Error:", error.code, error.message);
          });
  }
  function signup() {
      let e = email.value;
      let p = password.value;
      firebase.auth().createUserWithEmailAndPassword(e, p)
          .then((userCredential) => {
              // Successfully created user
              let user = userCredential.user;
          })
          .catch((error) => {
              console.error("Sign Up Error:", error.code, error.message);
          });
  }

  function logout() {
      firebase.auth().signOut().then(() => {
          // Sign-out successful
      }).catch((error) => {
          console.error("Logout Error:", error.code, error.message);
      });
      console.log('Logout button clicked!');
  }

  firebase.auth().onAuthStateChanged((user) => {
      const popupContainer = document.getElementById('popup-container');

      if (user) {
          fetch('loggedin.html')
              .then(response => response.text())
              .then(html => {
                  popupContainer.innerHTML = html;
                  setupEventListeners();
              })
              .catch(error => {
                  console.error('Error loading loggedin.html:', error);
              });
      } else {
          fetch('main.html')
              .then(response => response.text())
              .then(html => {
                  popupContainer.innerHTML = html;
                  setupEventListeners();
              })
              .catch(error => {
                  console.error('Error loading main.html:', error);
              });
      }
  });
});

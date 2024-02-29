// import {getDatabase,ref,set} from "firebase/database";

const firebaseConfig = {
  apiKey: "AIzaSyC19_MpWdcLnCyktjcpN7O_R3eSYGIbvgo",
  authDomain: "vocabhub-6f20f.firebaseapp.com",
  projectId: "vocabhub-6f20f",
  storageBucket: "vocabhub-6f20f.appspot.com",
  messagingSenderId: "990325792011",
  appId: "1:990325792011:web:1315867b935ad9e3bf560e",
};

// Initialize Firebase
try{
  firebase.initializeApp(firebaseConfig);
}
catch(e){
  console.log(e);
}

// chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
//   if (changeInfo.status === 'complete') {
//     // Get the current URL
//     const currentUrl = tab.url;

//     // Retrieve the stored URLs
//     chrome.storage.sync.get(['urlList'], function(result) {
//       const urlList = result.urlList || [];

//       // Add the current URL to the list
//       urlList.push(currentUrl);

//       // Save the updated list
//       chrome.storage.sync.set({ 'urlList': urlList });
//     });
//   }
// });


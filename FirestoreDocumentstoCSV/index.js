var fs = require('fs');

var admin = require("firebase-admin");
var serviceAccount = require("./serviceAccountKey.json");
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://datastore-9fd58.firebaseio.com"
});
var db = admin.firestore();
var fingerprintsRef = db.collection('fingerprints');


const Json2csvParser = require('json2csv').Parser;
const parser = new Json2csvParser();


var features = new Set();
var filter = new Set(['CDOT-MSRIT', 'Dr.MKN', 'CSE-LAB']);

var collection = [];

var count = 0;

var query = fingerprintsRef.get()
.then(snapshot => {
  snapshot.forEach(doc => {
        // console.log(doc.id, '=>', doc.data());
        var id = doc.id;
        var docData = doc.data();
        var roomId = docData.roomId;
        docData.data.forEach( item => {
          count++;
          var entry = {};
          entry['roomId'] = roomId;
          item.fingerprint.forEach(fingerprint => {
            if((filter.has('*') || filter.has(fingerprint.ssid)) && !features.has(fingerprint.bssid)){
              features.add(fingerprint.bssid);
              // entry[fingerprint.bssid] = fingerprint.level;
            }
            entry[fingerprint.bssid] = fingerprint.level;
          });
          collection.push(entry);
        });
      });
      //console.log(features.size);
      //console.log(features);
      console.log(count);
      //console.log(collection.slice(Math.max(collection.length - 5, 1)));
      //console.log(parser.parse(collection.slice(Math.max(collection.length - 5, 1))));
      fs.writeFile("data.csv", parser.parse(collection), function(err) {
        if(err) {
          return console.log(err);
        }
        console.log("The file was saved!");
      }); 
    })
.catch(err => {
  console.log('Error getting documents', err);
});
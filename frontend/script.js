const BASE_URL = "http://127.0.0.1:5000"; // Flask backend

let supportData = {}; // will hold category → example texts

// --- Step 2: Train model (build support set) ---
document.getElementById("train-btn").addEventListener("click", () => {
  const categoriesInput = document.getElementById("categories").value.trim();
  const examplesInput = document.getElementById("examples").value.trim();

  if (!categoriesInput || !examplesInput) {
    alert("Please enter both categories and example texts.");
    return;
  }

  // Parse categories
  const categories = categoriesInput.split(",").map(c => c.trim());
  supportData = {};
  categories.forEach(cat => (supportData[cat] = []));

  // Parse example lines
  const lines = examplesInput.split("\n").map(l => l.trim()).filter(l => l);
  lines.forEach(line => {
    const [cat, ...rest] = line.split(":");
    if (cat && rest.length > 0 && supportData[cat.trim()] !== undefined) {
      supportData[cat.trim()].push(rest.join(":").trim());
    }
  });

  alert("✅ Support data prepared! You can now test a new conversation.");
});


// --- Step 3: Predict ---
document.getElementById("predict-btn").addEventListener("click", async () => {
  const queryText = document.getElementById("test-text").value.trim();
  if (!queryText) {
    alert("Please enter a conversation to classify.");
    return;
  }

  if (Object.keys(supportData).length === 0) {
    alert("Please train the model first (Step 2).");
    return;
  }

  document.getElementById("prediction").innerText = "⏳ Classifying...";

  try {
    const res = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        support: supportData,
        query: queryText,
        inner_steps: 10,
        lr: 0.001
      })
    });

    const data = await res.json();

    if (data.error) {
      document.getElementById("prediction").innerText = `❌ ${data.error}`;
    } else {
      document.getElementById("prediction").innerText =
        `Predicted Category: ${data.category}\nConfidence: ${data.confidence}%`;
    }
  } catch (err) {
    document.getElementById("prediction").innerText =
      "❌ Error connecting to backend.";
    console.error(err);
  }
});

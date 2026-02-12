// ==UserScript==
// @name         Novel TTS (Zen Mode)
// @match        https://novelbin.com/b/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// ==/UserScript==

(function() {
    'use strict';

    // --- UI CONSTRUCTION ---
    const container = document.createElement('div');
    container.style = `
        position: fixed; bottom: 20px; right: 20px; z-index: 9999;
        background: #1a1a1a; padding: 15px; border-radius: 12px;
        color: #e0e0e0; font-family: 'Segoe UI', sans-serif;
        width: 320px; box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        border: 1px solid #333; display: flex; flex-direction: column; gap: 10px;
    `;
    document.body.appendChild(container);

    // Header / Status
    const status = document.createElement('div');
    status.innerText = 'Zen TTS Ready';
    status.style = 'font-size: 13px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;';
    container.appendChild(status);

    // Current Text Display
    const textPreview = document.createElement('div');
    textPreview.style = 'font-size: 14px; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; height: 20px;';
    container.appendChild(textPreview);

    // Play/Pause Button
    const btn = document.createElement('button');
    btn.innerText = '▶️ Start';
    btn.style = 'padding: 10px; cursor: pointer; background: #2d8cf0; color: white; border: none; border-radius: 6px; font-weight: bold; font-size: 14px; transition: background 0.2s;';
    container.appendChild(btn);

    // Controls Row
    const controlsRow = document.createElement('div');
    controlsRow.style = 'display: flex; align-items: center; gap: 10px; margin-top: 5px;';
    container.appendChild(controlsRow);

    // Speed Label
    const speedLabel = document.createElement('span');
    speedLabel.innerText = '1.0x';
    speedLabel.style = 'font-size: 12px; width: 30px; text-align: right;';

    // Speed Slider
    const speedSlider = document.createElement('input');
    speedSlider.type = 'range';
    speedSlider.min = '0.5';
    speedSlider.max = '2.0';
    speedSlider.step = '0.1';
    speedSlider.value = '1.0';
    speedSlider.style = 'flex-grow: 1; cursor: pointer; height: 4px;';

    speedSlider.oninput = () => {
        speedLabel.innerText = `${speedSlider.value}x`;
        // Clear cache so next fetch uses new speed
        clearFutureCache();
    };

    controlsRow.appendChild(document.createTextNode('Speed: '));
    controlsRow.appendChild(speedSlider);
    controlsRow.appendChild(speedLabel);

    // Progress Bar
    const progressContainer = document.createElement('div');
    progressContainer.style = 'height: 4px; background: #333; border-radius: 2px; overflow: hidden; margin-top: 5px;';
    const progressBar = document.createElement('div');
    progressBar.style = 'height: 100%; width: 0%; background: #2d8cf0; transition: width 0.3s;';
    progressContainer.appendChild(progressBar);
    container.appendChild(progressContainer);

    // --- STATE & AUDIO LOGIC ---
    let paragraphs = [];
    let currentIndex = 0;
    let isPlaying = false;
    let audioCache = {};
    let activeAudio = new Audio();
    const BUFFER_SIZE = 3;

    function parseContent() {
        // NovelBin specific cleaner
        const el = document.querySelector('#chr-content');
        if(!el) return [];
        return el.innerText.split('\n')
            .map(p => p.trim())
            .filter(p => p.length > 1); // Filter empty lines
    }

    async function fetchAudio(index) {
        if (index >= paragraphs.length || audioCache[index]) return;

        const text = paragraphs[index];
        const currentSpeed = parseFloat(speedSlider.value); // Get current speed value

        return new Promise((resolve) => {
            GM_xmlhttpRequest({
                method: "POST",
                url: "http://127.0.0.1:5000",
                data: JSON.stringify({ text: text, speed: currentSpeed }), // Send speed
                headers: { "Content-Type": "application/json" },
                responseType: "blob",
                onload: (res) => {
                    if(res.status !== 200) { resolve(null); return; }
                    const url = URL.createObjectURL(res.response);
                    audioCache[index] = url;
                    resolve(url);
                },
                onerror: () => resolve(null)
            });
        });
    }

    function clearFutureCache() {
        // If user changes speed, we must delete future buffered segments
        // so they are re-fetched with the new speed.
        const nextIdx = currentIndex + 1;
        for (let i = nextIdx; i < paragraphs.length; i++) {
            if (audioCache[i]) {
                URL.revokeObjectURL(audioCache[i]);
                delete audioCache[i];
            } else {
                break; // Stop if we hit unbuffered territory
            }
        }
    }

    async function playNext() {
        if (currentIndex >= paragraphs.length) {
            status.innerText = "Chapter Complete";
            btn.innerText = "Finished";
            return;
        }

        // Buffer current if missing
        if (!audioCache[currentIndex]) {
            status.innerText = `Buffering...`;
            await fetchAudio(currentIndex);
        }

        if (audioCache[currentIndex]) {
            activeAudio.src = audioCache[currentIndex];

            // Visual Update
            textPreview.innerText = paragraphs[currentIndex];
            status.innerText = `Playing ${currentIndex + 1}/${paragraphs.length}`;
            progressBar.style.width = `${((currentIndex + 1) / paragraphs.length) * 100}%`;

            activeAudio.play().catch(e => console.error("Play error", e));

            // Lazy Load Future
            for (let i = 1; i <= BUFFER_SIZE; i++) {
                fetchAudio(currentIndex + i);
            }
        }
    }

    activeAudio.onended = () => {
        if(audioCache[currentIndex]) {
            URL.revokeObjectURL(audioCache[currentIndex]);
            delete audioCache[currentIndex];
        }
        currentIndex++;
        playNext();
    };

    btn.onclick = () => {
        if (isPlaying) {
            activeAudio.pause();
            isPlaying = false;
            btn.innerText = '▶️ Resume';
            btn.style.background = '#444';
        } else {
            if (paragraphs.length === 0) {
                paragraphs = parseContent();
                currentIndex = 0;
            }
            isPlaying = true;
            btn.innerText = '⏸️ Pause';
            btn.style.background = '#2d8cf0';
            if (activeAudio.src && !activeAudio.ended) activeAudio.play();
            else playNext();
        }
    };
})();
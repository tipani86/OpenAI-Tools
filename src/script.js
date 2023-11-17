const streamlitDoc = window.parent.document

// Audio autoplay block override on mobile browsers (general idea from https://stackoverflow.com/questions/13266474/autoplay-audio-on-mobile-safari)

streamlitDoc.voicePlayer = document.getElementById('voicePlayer')
streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        streamlitDoc.voicePlayer.load()
        // Remove event listener after first keydown
        streamlitDoc.removeEventListener('keydown', function(e) {})
    }
})
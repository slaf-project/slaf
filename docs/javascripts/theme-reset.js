// Force light theme and clear cached preferences
(function() {
    // Clear any cached theme preference
    if (typeof localStorage !== 'undefined') {
        localStorage.removeItem('__palette');
        localStorage.removeItem('__palette_preferred');
    }

    // Force light theme on page load
    document.addEventListener('DOMContentLoaded', function() {
        // Set light theme as default
        const palette = __md_get("__palette");
        if (palette && palette.color) {
            palette.color.scheme = "default";
            __md_set("__palette", palette);
        }
    });
})();

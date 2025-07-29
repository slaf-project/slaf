// Theme-aware asset switching for SLAF documentation

(function() {
  'use strict';

  // Function to set favicon with correct path
  function setFavicon(href) {
    // Remove existing favicon links
    const existingFavicons = document.querySelectorAll('link[rel="icon"], link[rel="shortcut icon"], link[rel="apple-touch-icon"]');
    existingFavicons.forEach(link => link.remove());

    // Create SVG favicon (primary)
    const svgFavicon = document.createElement('link');
    svgFavicon.rel = 'icon';
    svgFavicon.type = 'image/svg+xml';
    svgFavicon.href = href;
    document.head.appendChild(svgFavicon);

    // Create fallback favicon for older browsers
    const fallbackFavicon = document.createElement('link');
    fallbackFavicon.rel = 'icon';
    fallbackFavicon.type = 'image/svg+xml';
    fallbackFavicon.href = href;
    document.head.appendChild(fallbackFavicon);

    // Create shortcut icon for maximum compatibility
    const shortcutIcon = document.createElement('link');
    shortcutIcon.rel = 'shortcut icon';
    shortcutIcon.type = 'image/svg+xml';
    shortcutIcon.href = href;
    document.head.appendChild(shortcutIcon);
  }

  // Function to detect current theme
  function getCurrentTheme() {
    const html = document.documentElement;
    const body = document.body;

    // Check all elements for theme indicators
    const allElements = document.querySelectorAll('*');
    const themeElements = Array.from(allElements).filter(el =>
      el.hasAttribute('data-md-color-scheme') ||
      el.classList.contains('md-color-scheme--slate') ||
      el.classList.contains('md-color-scheme--default')
    );

    // Method 1: Check data-md-color-scheme attribute on any element
    for (const element of themeElements) {
      if (element.hasAttribute('data-md-color-scheme')) {
        const scheme = element.getAttribute('data-md-color-scheme');
        return scheme === 'slate' ? 'dark' : 'light';
      }
    }

    // Method 2: Check for dark theme classes
    for (const element of themeElements) {
      if (element.classList.contains('md-color-scheme--slate')) {
        return 'dark';
      }
    }

    // Method 3: Check for light theme classes
    for (const element of themeElements) {
      if (element.classList.contains('md-color-scheme--default')) {
        return 'light';
      }
    }

    // Method 4: Check localStorage for theme preference
    const storedTheme = localStorage.getItem('__palette');
    if (storedTheme) {
      return storedTheme === 'slate' ? 'dark' : 'light';
    }

    return 'light';
  }

  // Function to update favicon based on theme
  function updateFavicon() {
    // Always use dark monochrome for favicon - no theme switching
    const faviconPath = 'assets/slaf-icon-transparent-dark-mono.svg';
    setFavicon(faviconPath);
  }

  // Function to update header icon based on theme
  function updateHeaderIcon() {
    // Always use dark monochrome for header icon - no theme switching
    const headerIcon = document.querySelector('.md-header__button.md-logo img');

    if (headerIcon) {
      const iconPath = 'assets/slaf-icon-transparent-dark-mono.svg';
      headerIcon.src = iconPath;
    }
  }

  // Update assets when theme changes
  function handleThemeChange() {
    updateFavicon();
    updateHeaderIcon();
  }

  // Expose function for manual testing (if needed)
  window.updateSLAFAssets = handleThemeChange;

  // Listen for theme changes
  document.addEventListener('DOMContentLoaded', function() {
    // Initial update
    handleThemeChange();

    // Watch for theme toggle clicks
    const themeToggle = document.querySelector('[data-md-toggle="palette"]');
    if (themeToggle) {
      themeToggle.addEventListener('click', function() {
        // Small delay to allow theme to change
        setTimeout(handleThemeChange, 100);
      });
    }

    // Listen for palette changes
    const observer = new MutationObserver(function(mutations) {
      let themeChanged = false;
      mutations.forEach(function(mutation) {
        if (mutation.type === 'attributes' &&
            (mutation.attributeName === 'data-md-color-scheme' ||
             mutation.attributeName === 'class')) {
          themeChanged = true;
        }
      });
      if (themeChanged) {
        setTimeout(handleThemeChange, 50);
      }
    });

    // Start observing document element, body, and any element with theme classes
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme', 'class']
    });

    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['class']
    });

    // Also observe any existing elements with theme indicators
    const themeElements = document.querySelectorAll('[data-md-color-scheme], .md-color-scheme--slate, .md-color-scheme--default');
    themeElements.forEach(element => {
      observer.observe(element, {
        attributes: true,
        attributeFilter: ['data-md-color-scheme', 'class']
      });
    });

    // Also add a periodic check for theme changes (fallback)
    setInterval(function() {
      const currentTheme = getCurrentTheme();
      if (window.lastKnownTheme !== currentTheme) {
        window.lastKnownTheme = currentTheme;
        handleThemeChange();
      }
    }, 1000);
  });
})();

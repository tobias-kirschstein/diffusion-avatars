
$(document).ready(function() {

    var carousels = bulmaCarousel.attach('#results-carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    });

    // Start playing next video in carousel and pause previous video to limit load on browser
    for(var i = 0; i < carousels.length; i++) {
        // Add listener to  event
        carousels[i].on('before:show', state => {
            var nextId = (state.next + state.length) % state.length;  // state.next can be -1 or larger than the number of videos
            var nextVideoElement = $("#results-carousel .slider-item[data-slider-index='" + nextId + "'] video")[0];
            var previousVideoElement = $("#results-carousel .slider-item[data-slider-index='" + state.index + "'] video")[0];

            previousVideoElement.pause();
            previousVideoElement.currentTime = 0;
            nextVideoElement.currentTime = 0;
            nextVideoElement.play();
        });
    }

    var reenactmentsCarousels = bulmaCarousel.attach('#reenactments-carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    });

    // Start playing next video in carousel and pause previous video to limit load on browser
    for(var i = 0; i < reenactmentsCarousels.length; i++) {
        // Add listener to  event
        reenactmentsCarousels[i].on('before:show', state => {
            var nextId = (state.next + state.length) % state.length;  // state.next can be -1 or larger than the number of videos
            var nextVideoElement = $("#reenactments-carousel .slider-item[data-slider-index='" + nextId + "'] video")[0];
            var previousVideoElement = $("#reenactments-carousel .slider-item[data-slider-index='" + state.index + "'] video")[0];

            previousVideoElement.pause();
            previousVideoElement.currentTime = 0;
            nextVideoElement.currentTime = 0;
            nextVideoElement.play();
        });
    }

})

EXPRESSION_ANIMATION = "37_from_76"

function updateAnimationMatrix(point) {
    const skip = 2;
    const n = 20 - skip;

    let top = Math.round((n / skip) * point.y.clamp(0, 1)) * skip;
    let left = Math.round((n / skip) * point.x.clamp(0, 1)) * skip;
    top = ("0000" + top).slice(-5);
    left = ("0000" + left).slice(-5);
    console.log(left, top);
    // $('.animation-matrix-rgb > img').css('left', -left + '%');
    // $('.animation-matrix-rgb > img').css('top', -top + '%');
    $('.animation-matrix-rgb > img').attr("src", "./static/images/matrix_animation/" + EXPRESSION_ANIMATION + "/frame_" + left + "_" + top + ".jpg");
    console.log($('.animation-matrix-rgb > img').attr("src"));
}


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

    const position = { x: 0, y: 0 }
    const box = $('.animation-matrix');

    Number.prototype.clamp = function(min, max) {
        return Math.min(Math.max(this, min), max);
    };

    const cursor = $('.animation-matrix-cursor');
    interact('.animation-matrix-cursor').draggable({
        listeners: {
            start (event) {
                console.log(event.type, event.target)
            },
            move (event) {
                position.x += event.dx
                position.y += event.dy

                event.target.style.transform =
                    `translate(${position.x}px, ${position.y}px)`

                let childPos = cursor.offset();
                let parentPos = box.offset();
                let childSize = cursor.outerWidth();
                let point = {
                    x: (childPos.left - parentPos.left),
                    y: (childPos.top - parentPos.top)
                };
                point = {
                    x: (point.x) / (box.innerWidth() - childSize),
                    y: (point.y) / (box.innerHeight() - childSize)
                }
                updateAnimationMatrix(point);
            },
        },
        modifiers: [
            interact.modifiers.restrictRect({
                restriction: 'parent'
            })
        ]
    });

})
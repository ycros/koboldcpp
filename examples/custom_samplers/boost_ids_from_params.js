var boosts = {};

function init(params) {
    boosts = (params && params.boosts) || {};
}

function sample(s) {
    for (var i = 0; i < s.size(); i++) {
        var key = String(s.id(i));
        if (boosts.hasOwnProperty(key)) {
            s.addLogit(i, boosts[key]);
        }
    }

    return s.pickGreedy();
}

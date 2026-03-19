function sample(s) {
    var recent = Math.min(64, s.recentCount());

    for (var i = 0; i < s.size(); i++) {
        var tokenId = s.id(i);
        for (var j = 0; j < recent; j++) {
            if (tokenId === s.recentToken(j)) {
                s.addLogit(i, -0.8);
                break;
            }
        }
    }

    return s.pick();
}

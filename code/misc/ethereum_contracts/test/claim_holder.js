const ClaimHolder = artifacts.require("./ClaimHolder.sol");

contract('ClaimHolder', function (accounts) {
    let instance;

    beforeEach('setup', async function () {
        instance = await ClaimHolder.new();
    });

    it('add a claim', async function () {
        let result = await instance.addClaim(1, 1, 1, 1, 1, "http://google.com");
        console.log(result);
    });
});

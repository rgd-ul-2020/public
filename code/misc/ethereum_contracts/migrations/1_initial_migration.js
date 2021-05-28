var Migrations  = artifacts.require("./Migrations.sol");
var ClaimHolder = artifacts.require("./ClaimHolder.sol");

module.exports = function(deployer) {
    deployer.deploy(Migrations);
    deployer.deploy(ClaimHolder);
};

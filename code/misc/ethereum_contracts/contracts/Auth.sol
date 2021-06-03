pragma solidity ^0.4.21;

contract Auth
{
/*

    function (address claim_holder_id) returns (bool)
    {
        ClaimHolder holder = ClaimHolder(claim_holder_id);
                          
        bytes32 claimId;
        uint256 claimType;
        uint256 scheme;
        address issuer;

        5     struct Claim {                                                                                  
      56         uint256 claimType;                                                                          
  57         uint256 scheme;                                                                             
  58         address issuer;    // msg.sender                                                            
  59         bytes   signature; // this.address + claimType + data                                       
  60         bytes   data;                                                                               
  61         string  uri;   
    }
*/
    event LOG(uint256);

    function issueClaim(address holder, address claimer, address publicKey)
        returns (bytes32 claim)
    {
        ClaimHolder claimHolder = ClaimHolder(holder);

        uint256 claimRequest = claimHolder.addClaim(3, 0, msg.sender, 0, 0, "https://google.com");
        emit LOG(uint256);
    }

    function removeClaim(address holder, address claimer, address publicKey, bytes32 claim)
        returns (bool success)
    {
    }

    function checkClaim(address holder, address claimer, address publicKey, bytes32 claim)
        returns (bool valid)
    {
    }
}

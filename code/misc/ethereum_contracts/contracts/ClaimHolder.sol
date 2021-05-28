pragma solidity ^0.4.21; 

import "./ERC735.sol";

contract ClaimHolder is ERC735
{                      
    mapping(bytes32 => Claim)     claims;
    mapping(uint256 => bytes32[]) claimIdsByType;

    function getClaim(bytes32 _claimId) public view
        returns (uint256 claimType, uint256 scheme,address issuer, 
            bytes signature, bytes data, string uri)
    {
        Claim storage claim = claims[_claimId];

        return (claim.claimType, claim.scheme, claim.issuer, claim.signature,
            claim.data, claim.uri);
    }

    function getClaimIdsByType(uint256 _claimType) public view
        returns (bytes32[] claimIds)
    {
        return claimIdsByType[_claimType];
    }

    function addClaim(uint256 _claimType, uint256 _scheme, address issuer,
        bytes _signature, bytes _data, string  _uri) public
        returns (uint256 claimRequestId)
    {
        bytes32 claimId = keccak256(uint256(issuer) + _claimType);

        if (claims[claimId].issuer != 0) {
            return 0;
        }

        Claim memory claim = Claim({
            claimType: _claimType,
            scheme:    _scheme,
            issuer:    issuer,
            signature: _signature,
            data:      _data,
            uri:       _uri
        });

        claims[claimId] = claim;
        claimIdsByType[_claimType].push(claimId);

        emit ClaimAdded(claimId, _claimType, _scheme, issuer, _signature, _data, _uri);

        return 1;
    }

    function removeClaim(bytes32 _claimId) public
        returns (bool success)
    {
        Claim storage claim = claims[_claimId];

        if (claim.issuer == 0) {
            return false;
        }

        bytes32[] storage claimIds = claimIdsByType[claim.claimType];

        for (uint256 i = 0; i < claimIds.length; i++) {
            if (claimIds[i] == _claimId) {
                delete claimIds[i];
                break;
            }
        }

        delete claims[_claimId];

        emit ClaimRemoved(_claimId, claim.claimType, claim.scheme, claim.issuer,
            claim.signature, claim.data, claim.uri);

        return true;
    }
}


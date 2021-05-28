package ie.lero.proto.tasks;

import android.os.AsyncTask;
import android.util.Log;

import java.io.IOException;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;

import ie.lero.proto.Network;

public class InetTask extends AsyncTask<Void, Void, Void>
{
    private Network network;

    public InetTask(Network network)
    {
        this.network = network;
    }

    @Override
    protected Void doInBackground(Void... voids)
    {
        try {
            DatagramSocket socket = new DatagramSocket();
            socket.connect(InetAddress.getByName("google.com"), 80);
            network.setLocalAddress((InetSocketAddress) socket.getLocalSocketAddress());
            socket.close();
        } catch (IOException e) {
            Log.e("INET_TASK", "Local Address Exception", e);
        }

        return null;
    }
}

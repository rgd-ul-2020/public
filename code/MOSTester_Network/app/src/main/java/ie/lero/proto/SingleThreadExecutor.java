package ie.lero.proto;

import android.support.annotation.NonNull;

import java.util.HashMap;
import java.util.concurrent.Executor;

public class SingleThreadExecutor implements Executor
{
    static
    {
        map = new HashMap<>();
    }

    private static HashMap<String, SingleThreadExecutor> map;

    public static SingleThreadExecutor getInstance(String key)
    {
        SingleThreadExecutor executor = map.get(key);

        if (executor == null) {
            executor = new SingleThreadExecutor(map.size());
            map.put(key, executor);
        }

        return executor;
    }

    private int    id;
    private Thread thread;

    private SingleThreadExecutor(int id)
    {
        this.id = id;
        thread = null;
    }

    @Override
    public void execute(@NonNull Runnable runnable)
    {
        if (thread != null) {
            thread.interrupt();
        }

        thread = new Thread(runnable);
        thread.setName("Thread #" + id);
        thread.start();
    }
}

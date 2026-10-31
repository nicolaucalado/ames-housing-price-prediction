# Kubernetes test

I tested the deployment on a local kind cluster (kind v0.29.0, Docker Desktop).

```bash
kind create cluster --name ames
kind load docker-image ames-housing:1.0 --name ames
kubectl apply -f k8s/
kubectl rollout status deployment/ames-housing
```

Pods:

```
NAME                            READY   STATUS    RESTARTS   AGE
ames-housing-858cd9d8d9-7s597   1/1     Running   0          13s
ames-housing-858cd9d8d9-hftql   1/1     Running   0          13s
```

Service:

```
NAME           TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
ames-housing   NodePort   10.96.185.161   <none>        80:30414/TCP   13s
```

After `kubectl port-forward svc/ames-housing 9697:80`:

`GET /health`:

```json
{"status":"ok","model":"Ridge","n_features":79}
```

`POST /predict`:

```json
{"predicted_price":137999.66,"predicted_log_price":11.835,"currency":"USD"}
```

The prediction is the same as the local API and the Docker container, so the artifact is
consistent across the three environments.

To reproduce, run the commands above and take a screenshot of the Swagger UI at
`http://localhost:9697/docs` (or record a short video).
